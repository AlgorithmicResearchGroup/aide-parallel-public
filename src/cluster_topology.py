"""Shared local/cluster topology helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import os
import yaml


DEFAULT_RAY_PORT = 6379
DEFAULT_DASHBOARD_PORT = 8265
DEFAULT_OBJECT_STORE_MEMORY = 10_000_000_000
DEFAULT_SSH_USER = "ubuntu"


@dataclass(frozen=True)
class ClusterNode:
    host: str
    role: str
    cpus: int | None
    gpus: int
    node_ip: str | None = None


@dataclass(frozen=True)
class ClusterTopology:
    mode: str
    config_path: str | None
    ssh_user: str
    python_bin: str | None
    working_dir: str | None
    head_host: str | None
    head_ip: str | None
    ray_port: int
    dashboard_port: int
    object_store_memory: int
    nodes: tuple[ClusterNode, ...]

    @property
    def is_local(self) -> bool:
        return self.mode == "local"

    @property
    def worker_nodes(self) -> tuple[ClusterNode, ...]:
        return tuple(node for node in self.nodes if node.role == "worker")

    @property
    def head_node(self) -> ClusterNode | None:
        for node in self.nodes:
            if node.role == "head":
                return node
        return None


def _normalize_node(raw: dict[str, Any]) -> ClusterNode:
    host = str(raw.get("host", "")).strip()
    role = str(raw.get("role", "")).strip().lower()
    if not host:
        raise ValueError("Cluster nodes must define a host")
    if role not in {"head", "worker"}:
        raise ValueError(f"Unsupported node role '{role}' for host '{host}'")
    cpus = raw.get("cpus")
    cpus_value = int(cpus) if cpus is not None else None
    gpus = int(raw.get("gpus", 0))
    node_ip = raw.get("node_ip")
    return ClusterNode(
        host=host,
        role=role,
        cpus=cpus_value,
        gpus=gpus,
        node_ip=str(node_ip).strip() if node_ip else None,
    )


def load_cluster_topology(cluster_config: str | None) -> ClusterTopology | None:
    if not cluster_config:
        return None

    path = Path(cluster_config).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cluster config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mode = str(payload.get("mode", "cluster")).strip().lower()
    if mode not in {"local", "cluster"}:
        raise ValueError("Cluster config mode must be 'local' or 'cluster'")

    ray_cfg = payload.get("ray", {}) or {}
    env_cfg = payload.get("env", {}) or {}
    nodes = tuple(_normalize_node(item) for item in payload.get("nodes", []) or [])

    head_host = ray_cfg.get("head_host")
    head_ip = ray_cfg.get("head_ip")
    ray_port = int(ray_cfg.get("port", DEFAULT_RAY_PORT))
    dashboard_port = int(ray_cfg.get("dashboard_port", DEFAULT_DASHBOARD_PORT))
    object_store_memory = int(ray_cfg.get("object_store_memory", DEFAULT_OBJECT_STORE_MEMORY))

    topology = ClusterTopology(
        mode=mode,
        config_path=str(path),
        ssh_user=str(env_cfg.get("ssh_user") or payload.get("ssh_user") or os.getenv("AIDE_SSH_USER", DEFAULT_SSH_USER)),
        python_bin=env_cfg.get("python_bin"),
        working_dir=env_cfg.get("working_dir"),
        head_host=str(head_host).strip() if head_host else None,
        head_ip=str(head_ip).strip() if head_ip else None,
        ray_port=ray_port,
        dashboard_port=dashboard_port,
        object_store_memory=object_store_memory,
        nodes=nodes,
    )
    validate_cluster_topology(topology)
    return topology


def validate_cluster_topology(topology: ClusterTopology) -> ClusterTopology:
    if topology.mode == "local":
        return topology

    if not topology.nodes:
        raise ValueError("Cluster mode requires at least one declared node")
    heads = [node for node in topology.nodes if node.role == "head"]
    if len(heads) != 1:
        raise ValueError("Cluster mode requires exactly one head node")
    if not topology.head_ip:
        raise ValueError("Cluster mode requires ray.head_ip in the cluster config")
    if not topology.head_host:
        raise ValueError("Cluster mode requires ray.head_host in the cluster config")
    if not topology.worker_nodes:
        raise ValueError("Cluster mode requires at least one worker node")
    return topology


def resolve_execution_target(
    *,
    local_flag: bool = False,
    head_node_ip: str | None = None,
    cluster_config: str | None = None,
) -> tuple[bool, str | None, ClusterTopology | None]:
    topology = load_cluster_topology(cluster_config)
    if local_flag and topology and not topology.is_local:
        raise ValueError("Cannot combine --local with a cluster-mode topology config")
    if local_flag:
        return True, None, topology
    if topology:
        if topology.is_local:
            return True, None, topology
        return False, topology.head_ip, topology
    if head_node_ip:
        return False, head_node_ip, None
    return True, None, None


def run_host_is_local(host: str | None) -> bool:
    return host in {None, "", "localhost", "127.0.0.1"}


def topology_summary(topology: ClusterTopology | None) -> dict[str, Any]:
    if topology is None:
        return {"mode": "implicit_local"}
    return {
        "mode": topology.mode,
        "config_path": topology.config_path,
        "head_host": topology.head_host,
        "head_ip": topology.head_ip,
        "ray_port": topology.ray_port,
        "dashboard_port": topology.dashboard_port,
        "ssh_user": topology.ssh_user,
        "nodes": [
            {
                "host": node.host,
                "role": node.role,
                "cpus": node.cpus,
                "gpus": node.gpus,
                "node_ip": node.node_ip,
            }
            for node in topology.nodes
        ],
    }
