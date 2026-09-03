#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Triton 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测三种退化类型：
  Type 1: 无 @triton.jit kernel，全部使用 PyTorch
  Type 2: 有 @triton.jit kernel 定义但 forward() 未调用
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口

用法:
    python validate_triton_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import logging
import os
import sys


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402

logger = logging.getLogger("triton_op_verifier.validate_triton_impl")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
# ---------------------------------------------------------------------------

ALLOWED_TORCH_FUNCS = {
    # buffer 分配
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    # tensor 创建（有时需要用于标量常量 / 索引）
    "tensor", "arange", "linspace",
    # 类型 / 设备
    "as_tensor",
}

FORBIDDEN_FORWARD_TORCH_FUNCS = {
    "arange", "as_tensor", "cat", "linspace", "stack", "tensor",
}

FORBIDDEN_FORWARD_BUILTINS = {
    "dict", "max", "min", "range", "set", "sorted", "sum",
}

FORBIDDEN_KERNEL_BUILTINS = {
    "abs", "bool", "float", "int", "len", "max", "min",
}

ALLOWED_TENSOR_METHODS = {
    # 形状 / 元信息
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    # 布局操作（不执行计算）
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "cuda",
    # 原地标记
    "requires_grad_", "zero_",
    # 切片相关（一般通过 __getitem__ 而非方法，但以防万一）
    "index_select",
}

ALLOWED_TRITON_ATTRS = {
    "cdiv", "next_power_of_2",
}

FORBIDDEN_STATIC_KEYWORDS = {
    "enable_linearize",
    "force_simt_template",
    "num_buffers_warp_spec",
    "num_consumer_groups",
    "num_ctas",
    "num_stages",
    "num_warps",
    "reg_dec_producer",
    "reg_inc_consumer",
}

FORBIDDEN_TL_CALLS = {
    "constant": (
        "tl.constant(...) is not available on this Ascend Triton path. Use a "
        "plain Python literal for scalar constants, or cast tl values with "
        "value.to(...)."
    ),
    "constexpr": (
        "tl.constexpr is a type annotation for compile-time parameters, not a "
        "runtime function. Annotate kernel arguments as ARG: tl.constexpr "
        "instead of calling tl.constexpr(...)."
    ),
}

FORBIDDEN_TRITON_CALLS = {
    "get_device_properties": (
        "Do not query device properties inside generated implementations. Keep "
        "launch configuration static and derived from tensor shapes."
    ),
}

FORBIDDEN_TRITON_DECORATORS = {
    "autotune": (
        "Do not use triton.autotune in rollout implementations. Generate one "
        "deterministic correctness kernel first."
    ),
    "heuristics": (
        "Do not use triton.heuristics in rollout implementations. It has "
        "caused fragile key/config failures on this path."
    ),
}

FORBIDDEN_TENSOR_METHODS = {
    # 计算操作
    "sum", "mean", "max", "min", "softmax", "log_softmax",
    "matmul", "mm", "bmm", "addmm", "add", "sub", "mul", "div",
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "norm", "layer_norm", "batch_norm", "group_norm",
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    "dropout", "softplus", "hardtanh", "hardswish",
    # 设备到 host 的同步/转换
    "item", "tolist",
}

FUNCTIONAL_QUALIFIERS = {
    "F", "functional", "torch.nn.functional", "nn.functional",
}

# forward() 中禁止的 Python 控制流和结构
FORBIDDEN_PYTHON_STMTS = {
    "for": "Python for 循环",
    "while": "Python while 循环",
}


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------

def _decorator_is_triton_jit(decorator):
    """判断装饰器节点是否为 triton.jit 或 @jit（从 triton 导入）。"""
    # @triton.jit
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "jit"):
            return True
    # @jit（直接导入）
    if isinstance(decorator, ast.Name) and decorator.id == "jit":
        return True
    # @triton.jit 作为 Call（如 @triton.jit 带参数，虽然少见）
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_jit(decorator.func)
    return False


def _decorator_is_triton_autotune(decorator):
    """判断装饰器是否为 triton.autotune。"""
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "autotune"):
            return True
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_autotune(decorator.func)
    return False


def _has_triton_decorator(func_node):
    """检查函数是否有 @triton.jit（可能与 @triton.autotune 组合）。"""
    for dec in func_node.decorator_list:
        if _decorator_is_triton_jit(dec):
            return True
    return False


def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          my_func    -> (None, 'my_func')
          self.conv  -> ('self', 'conv')
          kernel[g]  -> 返回 None（kernel launch 通过 Subscript）
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return (func.value.id, func.attr)
        # 处理 torch.nn.functional.relu 形式
        if isinstance(func.value, ast.Attribute):
            inner = func.value
            if isinstance(inner.value, ast.Name):
                return (f"{inner.value.id}.{inner.attr}", func.attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


def _get_subscript_value_name(node):
    """从 kernel[grid](...) 的 Subscript 节点提取 kernel 名称。"""
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            return node.value.id
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name):
                return f"{node.value.value.id}.{node.value.attr}"
    return None


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------

def _kernel_uses_tl_api(func_node) -> bool:
    """判断 kernel 函数体内是否出现 tl.* 属性访问。"""
    for child in ast.walk(func_node):
        if isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name) and child.value.id == "tl":
                return True
    return False


def find_triton_kernels(tree):
    """查找所有 @triton.jit 装饰的函数名，及其是否使用了 tl.* API。"""
    kernels = {}  # name -> {"has_tl_usage": bool, "line": int}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node):
            kernels[node.name] = {
                "has_tl_usage": _kernel_uses_tl_api(node),
                "line": node.lineno,
            }
    return kernels


def _find_forward_in_class(class_node):
    """从 ModelNew 类节点中找到 forward 方法。"""
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
            return item
    return None


def find_model_new_forward(tree):
    """找到 ModelNew 类的 forward 方法节点。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            forward = _find_forward_in_class(node)
            if forward is not None:
                return forward
    return None


def _call_invokes_kernel(call_node, kernel_names) -> bool:
    """判断单个 Call 节点是否启动了 triton kernel（直接或通过 Subscript）。"""
    if isinstance(call_node.func, ast.Subscript):
        return _get_subscript_value_name(call_node.func) in kernel_names
    resolved = _resolve_call_name(call_node)
    return bool(resolved and resolved[0] is None and resolved[1] in kernel_names)


def _helper_function_nodes(tree, kernel_names):
    """Collect module helpers and ModelNew methods reachable from forward()."""
    helpers = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names:
            helpers[node.name] = node
        elif isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name != "forward"
                    and item.name not in kernel_names
                ):
                    helpers[item.name] = item
    return helpers


def _called_helper(call_node, helpers):
    resolved = _resolve_call_name(call_node)
    if not resolved:
        return None
    qual, attr = resolved
    if attr in helpers and qual in (None, "self"):
        return attr
    return None


def forward_execution_nodes(tree, forward_node, kernel_names):
    """Return forward() and every helper it reaches, in call-graph order."""
    helpers = _helper_function_nodes(tree, kernel_names)
    execution_nodes = [forward_node]
    seen = {id(forward_node)}
    pending = [forward_node]
    while pending:
        current = pending.pop()
        for child in ast.walk(current):
            if not isinstance(child, ast.Call):
                continue
            helper_name = _called_helper(child, helpers)
            if helper_name is None:
                continue
            helper = helpers[helper_name]
            if id(helper) in seen:
                continue
            seen.add(id(helper))
            execution_nodes.append(helper)
            pending.append(helper)
    return execution_nodes


def kernel_launches(execution_nodes, kernel_names):
    """Return all Triton kernels launched along the reachable forward path."""
    launches = []
    for func_node in execution_nodes:
        for child in ast.walk(func_node):
            if isinstance(child, ast.Call) and _call_invokes_kernel(child, kernel_names):
                launches.append(_get_subscript_value_name(child.func) or "<kernel>")
    return launches


# --- check_forbidden_torch_ops 拆分出的辅助规则 ---

def _violation_for_loop(node):
    """循环禁用规则：返回违规字典或 None。"""
    if isinstance(node, ast.For):
        return {
            "line": node.lineno,
            "call": "for 循环",
            "reason": "forward() 中禁止 Python for 循环，核心计算必须在单个 Triton kernel 内完成",
        }
    if isinstance(node, ast.While):
        return {
            "line": node.lineno,
            "call": "while 循环",
            "reason": (
                "forward() 中禁止 Python while 循环，"
                "核心计算必须在单个 Triton kernel 内完成"
            ),
        }
    return None


def _violation_matmul_op(node):
    """检测 @ 运算符。"""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return {
            "line": node.lineno,
            "call": "@",
            "reason": "矩阵乘法 @ 运算符必须在 Triton kernel 中实现",
        }
    return None


def _violation_list_append(node):
    """检测 list.append 形式调用。"""
    # Shape plumbing such as ``out_shape = list(x.shape); out_shape.append(...)``
    # is allowed in forward(). Treating every append as host-side computation made
    # agents rewrite otherwise valid implementations instead of fixing kernels.
    return None


def _violation_comprehension(node):
    if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
        return {
            "line": node.lineno,
            "call": type(node).__name__,
            "reason": (
                "forward() must not build dynamic Python collections with "
                "comprehensions. Allocate output tensors and do computation in "
                "Triton kernels."
            ),
        }
    return None


def _violation_for_torch_qual(node, qual, attr):
    """处理 qual == 'torch' 的调用。"""
    if attr in FORBIDDEN_FORWARD_TORCH_FUNCS:
        return {
            "line": node.lineno,
            "call": f"torch.{attr}",
            "reason": (
                f"torch.{attr} in forward() is treated as host-side tensor "
                "creation/computation. Keep forward() to output allocation, "
                "shape plumbing, and Triton kernel launches."
            ),
        }
    if attr in ALLOWED_TORCH_FUNCS:
        return None
    return {
        "line": node.lineno,
        "call": f"torch.{attr}",
        "reason": f"torch.{attr} 是计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_functional_qual(node, qual, attr):
    """处理 F./functional. 形式调用。"""
    return {
        "line": node.lineno,
        "call": f"{qual}.{attr}",
        "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_tensor_method(node, qual, attr):
    """处理被禁止的 tensor 方法调用。"""
    if attr not in FORBIDDEN_TENSOR_METHODS:
        return None
    # 排除已知安全的 qual（torch/F/triton 已在上面处理）
    skip_quals = {"torch", "F", "triton"} | FUNCTIONAL_QUALIFIERS
    if qual in skip_quals:
        return None
    return {
        "line": node.lineno,
        "call": f"{qual}.{attr}()" if qual else f"{attr}()",
        "reason": f"{attr} 是计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_self_call(node, qual, attr):
    """处理 self.xxx(...) 形式调用。"""
    if qual != "self" or attr == "forward":
        return None
    return {
        "line": node.lineno,
        "call": f"self.{attr}(...)",
        "reason": (
            f"self.{attr}() 疑似 nn.Module 前向调用，"
            "核心计算必须在 Triton kernel 中实现"
        ),
    }


def _violation_for_call(node):
    """对 Call 节点应用所有调用相关规则，返回首个命中或 None。"""
    v = _violation_list_append(node)
    if v is not None:
        return v

    # --- kernel launch: kernel[grid](...) —— 允许 ---
    if isinstance(node.func, ast.Subscript):
        return None

    resolved = _resolve_call_name(node)
    if resolved is None:
        return None

    qual, attr = resolved
    if qual is None and attr in FORBIDDEN_FORWARD_BUILTINS:
        return {
            "line": node.lineno,
            "call": f"{attr}(...)",
            "reason": (
                f"Python builtin {attr}(...) in forward() is treated as "
                "host-side computation. Move the logic into the Triton kernel "
                "or make it static shape plumbing outside the hot path."
            ),
        }
    if qual == "torch":
        return _violation_for_torch_qual(node, qual, attr)
    if qual in FUNCTIONAL_QUALIFIERS:
        return _violation_for_functional_qual(node, qual, attr)
    # --- triton.cdiv 等 —— 允许 ---
    if qual == "triton" and attr in ALLOWED_TRITON_ATTRS:
        return None
    v = _violation_for_tensor_method(node, qual, attr)
    if v is not None:
        return v
    return _violation_for_self_call(node, qual, attr)


def _violation_for_node(node):
    """对任意 AST 节点应用所有规则，返回首个命中或 None。"""
    v = _violation_for_loop(node)
    if v is not None:
        return v
    v = _violation_matmul_op(node)
    if v is not None:
        return v
    v = _violation_comprehension(node)
    if v is not None:
        return v
    if isinstance(node, ast.Call):
        return _violation_for_call(node)
    return None


def check_forbidden_torch_ops(execution_nodes, kernel_names):
    """检查 forward 调用链是否使用了禁止的 host 计算。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]
    """
    violations = []
    if not execution_nodes:
        return violations

    for func_node in execution_nodes:
        for node in ast.walk(func_node):
            v = _violation_for_node(node)
            if v is not None:
                violations.append(v)

    return violations


def _violation_for_data_ptr(node):
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    _qual, attr = resolved
    if attr != "data_ptr":
        return None
    return {
        "line": node.lineno,
        "call": ".data_ptr()",
        "reason": (
            "Do not pass raw integer pointers to Triton kernels. Launch kernels "
            "with tensor objects directly, e.g. kernel[grid](x, out, ...)."
        ),
    }


def _violation_for_tl_to(node):
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    qual, attr = resolved
    if qual != "tl" or attr != "to":
        return None
    return {
        "line": node.lineno,
        "call": "tl.to(...)",
        "reason": (
            "triton.language has no tl.to API on this Ascend path. Cast tensor "
            "values with value.to(tl.float32) or another supported expression."
        ),
    }


def _violation_for_forbidden_tl_call(node):
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    qual, attr = resolved
    if qual != "tl" or attr not in FORBIDDEN_TL_CALLS:
        return None
    return {
        "line": node.lineno,
        "call": f"tl.{attr}(...)",
        "reason": FORBIDDEN_TL_CALLS[attr],
    }


def _violation_for_forbidden_triton_call(node):
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    qual, attr = resolved
    if qual != "triton" or attr not in FORBIDDEN_TRITON_CALLS:
        return None
    return {
        "line": node.lineno,
        "call": f"triton.{attr}(...)",
        "reason": FORBIDDEN_TRITON_CALLS[attr],
    }


def _violation_for_forbidden_keyword(node):
    if not isinstance(node, ast.Call):
        return None
    for keyword in node.keywords:
        if keyword.arg in FORBIDDEN_STATIC_KEYWORDS:
            return {
                "line": node.lineno,
                "call": f"{keyword.arg}=...",
                "reason": (
                    f"Do not tune '{keyword.arg}' in generated Triton Ascend "
                    "code. These tuning kwargs are rejected/noisy on this path."
                ),
            }
    return None


def _decorator_triton_attr(decorator):
    if isinstance(decorator, ast.Call):
        return _decorator_triton_attr(decorator.func)
    if not isinstance(decorator, ast.Attribute):
        return None
    if isinstance(decorator.value, ast.Name) and decorator.value.id == "triton":
        return decorator.attr
    return None


def _violation_for_forbidden_triton_decorator(node):
    if not isinstance(node, ast.FunctionDef):
        return None
    for decorator in node.decorator_list:
        attr = _decorator_triton_attr(decorator)
        if attr in FORBIDDEN_TRITON_DECORATORS:
            return {
                "line": node.lineno,
                "call": f"@triton.{attr}",
                "reason": FORBIDDEN_TRITON_DECORATORS[attr],
            }
    return None


def _violation_for_raise_or_assert(node):
    if isinstance(node, ast.Assert):
        return {
            "line": node.lineno,
            "call": "assert",
            "reason": (
                "Do not gate verifier cases with Python assert. Handle the "
                "input shape/dtype or let the Triton kernel mask unsupported "
                "lanes."
            ),
        }
    if isinstance(node, ast.Raise):
        exc = node.exc
        name = None
        if isinstance(exc, ast.Call):
            resolved = _resolve_call_name(exc)
            if resolved:
                name = resolved[1]
        elif isinstance(exc, ast.Name):
            name = exc.id
        if name == "NotImplementedError":
            return {
                "line": node.lineno,
                "call": "raise NotImplementedError",
                "reason": (
                    "Do not leave unsupported verifier paths in ModelNew. "
                    "Implement the case or simplify the kernel instead."
                ),
            }
    return None


def _annotation_is_tl_constexpr(annotation):
    if isinstance(annotation, ast.Attribute):
        return (
            isinstance(annotation.value, ast.Name)
            and annotation.value.id == "tl"
            and annotation.attr == "constexpr"
        )
    if isinstance(annotation, ast.Name):
        return annotation.id == "constexpr"
    return False


def _kernel_static_names(func):
    names = set()
    for arg in func.args.args:
        if arg.annotation is not None and _annotation_is_tl_constexpr(arg.annotation):
            names.add(arg.arg)
    for node in ast.walk(func):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        iter_call = node.iter
        if not isinstance(iter_call, ast.Call):
            continue
        resolved = _resolve_call_name(iter_call)
        if resolved == ("tl", "static_range"):
            names.add(node.target.id)
    return names


def _is_static_kernel_condition(node, static_names=None):
    """Best-effort allowlist for compile-time-looking Triton conditions."""
    static_names = static_names or set()
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id.isupper() or node.id in static_names
    if isinstance(node, ast.UnaryOp):
        return _is_static_kernel_condition(node.operand, static_names)
    if isinstance(node, ast.Compare):
        names = [child.id for child in ast.walk(node) if isinstance(child, ast.Name)]
        return bool(names) and all(
            name.isupper() or name in static_names or name in {"tl", "triton"}
            for name in names
        )
    return False


def _range_args_are_static(node):
    return all(isinstance(arg, ast.Constant) for arg in node.args)


def _violation_for_kernel_break_continue(node):
    if isinstance(node, (ast.Break, ast.Continue)):
        return {
            "line": node.lineno,
            "call": type(node).__name__.lower(),
            "reason": (
                "break/continue inside @triton.jit is fragile on the Ascend "
                "path. Use masks and tl.where instead of Python loop control."
            ),
        }
    return None


def _violation_for_kernel_python_cast(node):
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    qual, attr = resolved
    if qual is None and attr in FORBIDDEN_KERNEL_BUILTINS:
        return {
            "line": node.lineno,
            "call": f"{attr}(...)",
            "reason": (
                f"Python {attr}(...) inside @triton.jit often receives tl "
                "values and fails to compile. Use value.to(...), tl.where, "
                "or compile-time constexpr branches."
            ),
        }
    return None


def _violation_for_kernel_range(node):
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    qual, attr = resolved
    if qual is None and attr == "range" and not _range_args_are_static(node):
        return {
            "line": node.lineno,
            "call": "range(...)",
            "reason": (
                "Dynamic Python range(...) inside @triton.jit is unsupported "
                "or fragile. Use tl.arange/masks, or tl.static_range for true "
                "compile-time loops."
            ),
        }
    return None


def _violation_for_kernel_boolop(node):
    if isinstance(node, ast.BoolOp):
        return {
            "line": node.lineno,
            "call": "and/or",
            "reason": (
                "Python boolean and/or inside @triton.jit is invalid for tl "
                "tensor values. Combine masks with &, |, and tl.where."
            ),
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        if isinstance(node.left, ast.BinOp) or isinstance(node.right, ast.BinOp):
            return {
                "line": node.lineno,
                "call": "&/| chain",
                "reason": (
                    "Chained mask expressions like a & b & c have triggered "
                    "Ascend compiler failures. Split masks into named steps."
                ),
            }
    return None


def _violation_for_kernel_chained_compare(node):
    if isinstance(node, ast.Compare) and len(node.ops) > 1:
        return {
            "line": node.lineno,
            "call": "chained comparison",
            "reason": (
                "Chained comparisons such as a < b < c are unsupported or "
                "fragile in @triton.jit on this path. Split them into named "
                "boolean masks and combine with &."
            ),
        }
    return None


def _violation_for_kernel_dynamic_if(node, static_names=None):
    if isinstance(node, ast.If) and not _is_static_kernel_condition(node.test, static_names):
        return {
            "line": node.lineno,
            "call": "if ...",
            "reason": (
                "Runtime Python if inside @triton.jit is fragile on Ascend. "
                "Use tl.where/masks, or branch only on true compile-time "
                "conditions."
            ),
        }
    if isinstance(node, ast.IfExp) and not _is_static_kernel_condition(node.test, static_names):
        return {
            "line": node.lineno,
            "call": "x if cond else y",
            "reason": (
                "Python conditional expressions inside @triton.jit should be "
                "replaced with tl.where or compile-time constexpr branches."
            ),
        }
    return None


def _kernel_static_violations(tree):
    violations = []
    checks = (
        _violation_for_kernel_break_continue,
        _violation_for_kernel_python_cast,
        _violation_for_kernel_range,
        _violation_for_kernel_boolop,
        _violation_for_kernel_chained_compare,
        _violation_for_kernel_dynamic_if,
    )
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef) or not _has_triton_decorator(func):
            continue
        static_names = _kernel_static_names(func)
        for node in ast.walk(func):
            for check in checks:
                if check is _violation_for_kernel_dynamic_if:
                    violation = check(node, static_names)
                else:
                    violation = check(node)
                if violation is not None:
                    violation = dict(violation)
                    violation["call"] = f"{func.name}: {violation['call']}"
                    violations.append(violation)
                    break
    return violations


def check_static_illegal_patterns(tree):
    """Catch known unsupported generated-code patterns before NPU verify."""
    violations = []
    for node in ast.walk(tree):
        for check in (
            _violation_for_data_ptr,
            _violation_for_tl_to,
            _violation_for_forbidden_tl_call,
            _violation_for_forbidden_triton_call,
            _violation_for_forbidden_keyword,
            _violation_for_forbidden_triton_decorator,
            _violation_for_raise_or_assert,
        ):
            violation = check(node)
            if violation is not None:
                violations.append(violation)
                break
    violations.extend(_kernel_static_violations(tree))
    return violations


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def _empty_result(filepath):
    return {
        "valid": False,
        "filepath": filepath,
        "checks": {
            "triton_kernel_exists": {"passed": False, "kernels": [], "error": None},
            "kernel_called_from_forward": {"passed": False, "called": [], "error": None},
            "no_forbidden_torch_ops": {"passed": False, "violations": [], "error": None},
            "no_static_illegal_patterns": {"passed": False, "violations": [], "error": None},
        },
        "regression_type": None,
        "suggestion": "",
    }


def _check_kernel_exists(result, tree):
    """填充 triton_kernel_exists 检查；返回 (passed, kernel_names)。"""
    kernels = find_triton_kernels(tree)
    kernel_names = set(kernels.keys())
    result["checks"]["triton_kernel_exists"]["kernels"] = [
        {"name": k, "line": v["line"], "has_tl_usage": v["has_tl_usage"]}
        for k, v in kernels.items()
    ]

    if not kernel_names:
        result["checks"]["triton_kernel_exists"]["error"] = "未找到任何 @triton.jit 装饰的 kernel 函数"
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有 Triton kernel。必须创建至少一个 @triton.jit 装饰的函数，"
            "在其中使用 tl.load/tl.store 实现核心计算逻辑。"
        )
        return False, kernel_names

    kernels_without_tl = [k for k, v in kernels.items() if not v["has_tl_usage"]]
    if len(kernels_without_tl) == len(kernels):
        result["checks"]["triton_kernel_exists"]["error"] = (
            f"kernel 函数 {kernels_without_tl} 未使用任何 tl.* API，"
            "可能是空壳 kernel"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "虽然存在 @triton.jit 装饰的函数，但没有使用 triton.language (tl) API。"
            "kernel 必须使用 tl.load/tl.store 等进行显式内存操作和计算。"
        )
        return False, kernel_names

    result["checks"]["triton_kernel_exists"]["passed"] = True
    return True, kernel_names


def _check_forward_calls_kernel(result, tree, kernel_names):
    """填充 kernel_called_from_forward；返回可达的 forward 执行路径。"""
    forward_node = find_model_new_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew 类或 forward 方法。"
        return False, None

    execution_nodes = forward_execution_nodes(tree, forward_node, kernel_names)
    called = kernel_launches(execution_nodes, kernel_names)
    result["checks"]["kernel_called_from_forward"]["called"] = sorted(set(called))

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"@triton.jit kernel {list(kernel_names)} 已定义但 forward() 未调用任何 kernel"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"已定义 kernel {list(kernel_names)} 但 ModelNew.forward() 中未调用。"
            "forward() 必须通过 kernel_name[grid](...) 形式启动 kernel。"
        )
        return False, forward_node

    result["checks"]["kernel_called_from_forward"]["passed"] = True
    return True, execution_nodes


def _check_no_forbidden_ops(result, execution_nodes, kernel_names):
    """填充 no_forbidden_torch_ops 检查；返回 passed。"""
    violations = check_forbidden_torch_ops(execution_nodes, kernel_names)
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if not violations:
        result["checks"]["no_forbidden_torch_ops"]["passed"] = True
        return True

    result["checks"]["no_forbidden_torch_ops"]["error"] = (
        f"forward() 调用链中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"forward() 调用链启动了 Triton kernel 但仍使用 PyTorch 进行部分计算: "
        f"{violation_details}。"
        "所有核心计算必须在 @triton.jit kernel 中完成，"
        "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
    )
    return False


def _check_no_static_illegal_patterns(result, tree):
    """Fill no_static_illegal_patterns check and return passed."""
    violations = check_static_illegal_patterns(tree)
    result["checks"]["no_static_illegal_patterns"]["violations"] = violations

    if not violations:
        result["checks"]["no_static_illegal_patterns"]["passed"] = True
        return True

    result["checks"]["no_static_illegal_patterns"]["error"] = (
        f"Found {len(violations)} known unsupported Triton Ascend pattern(s)"
    )
    violation_details = "; ".join(
        f"line {v['line']}: {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        "Repair known unsupported patterns before running NPU verifier: "
        f"{violation_details}. "
        "Pass tensors directly to kernels; avoid tl.to(...), tl.constant(...), "
        "and tl.constexpr(...) calls; remove unsupported tuning/decorator "
        "configuration; avoid assert/NotImplementedError fallbacks and runtime "
        "Python control flow, casts, or chained comparisons inside @triton.jit."
    )
    return False


def validate(code, filepath="<unknown>"):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = _empty_result(filepath)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["triton_kernel_exists"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    ok, kernel_names = _check_kernel_exists(result, tree)
    if not ok:
        return result

    ok, execution_nodes = _check_forward_calls_kernel(result, tree, kernel_names)
    if not ok:
        return result

    if not _check_no_static_illegal_patterns(result, tree):
        return result

    if not _check_no_forbidden_ops(result, execution_nodes, kernel_names):
        return result

    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_code(path, want_json):
    """读取代码文件；失败时返回 None，由调用方决定退出。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        if want_json:
            logger.info("%s", json.dumps({"valid": False, "error": f"文件不存在: {path}"}))
        else:
            logger.error("[ERROR] 文件不存在: %s", path)
        return None


def _emit_pass(result):
    kernels = result["checks"]["triton_kernel_exists"]["kernels"]
    called = result["checks"]["kernel_called_from_forward"]["called"]
    logger.info("[PASS] Triton 实现验证通过")
    logger.info(
        "  - 发现 %d 个 @triton.jit kernel: %s",
        len(kernels),
        ", ".join(k["name"] for k in kernels),
    )
    logger.info("  - forward() 调用: %s", ", ".join(called))
    logger.info("  - forward() 中无禁止的 PyTorch 计算操作")


def _emit_fail(result):
    rtype = result["regression_type"]
    type_desc = {
        1: "完全无 Triton kernel（纯 PyTorch）",
        2: "有 Triton kernel 但 forward() 未调用",
        3: "部分计算使用 PyTorch（需全部移入 Triton kernel）",
    }
    logger.info("[FAIL] 检测到 PyTorch 退化 — Type %s: %s", rtype, type_desc.get(rtype, "未知"))

    for check_name, check_result in result["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        logger.info("  [%s] %s", status, check_name)
        if check_result["error"]:
            logger.info("         %s", check_result["error"])

    if result["checks"]["no_forbidden_torch_ops"]["violations"]:
        logger.info("  违规详情:")
        for v in result["checks"]["no_forbidden_torch_ops"]["violations"]:
            logger.info("    第 %s 行: %s — %s", v["line"], v["call"], v["reason"])

    logger.info("\n  修复建议: %s", result["suggestion"])


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(
        description="检查生成代码是否退化为 PyTorch 原生实现（AST 静态分析）"
    )
    parser.add_argument("file", help="要检查的 Python 文件路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    code = _load_code(args.file, args.json)
    if code is None:
        sys.exit(1)
    result = validate(code, filepath=args.file)

    if args.json:
        logger.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    elif result["valid"]:
        _emit_pass(result)
    else:
        _emit_fail(result)

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
