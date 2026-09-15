"""Recipe-only trajectory diagnostics using the configured Framework subclass."""

from uni_agent.framework.framework import GatewayAgentFramework
from uni_agent.gateway.session import Trajectory


class KernelAgentFramework(GatewayAgentFramework):
    def _trajectory_meta(self, traj: Trajectory) -> dict[str, object]:
        return {
            **super()._trajectory_meta(traj),
            "kernel_bench": traj.extra_fields.get("kernel_bench_diagnostics", {}),
        }
