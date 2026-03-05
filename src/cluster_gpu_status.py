#!/usr/bin/env python3
"""
GPU Status Monitor for Ray Cluster

This utility monitors GPU utilization across the Ray cluster, showing:
- GPU availability and allocation
- Memory usage per GPU
- Running experiments and their GPU assignments
- Cluster-wide resource status
"""

import ray
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any
import subprocess
import json
import os


def get_nvidia_smi_info(node_ip: str = None, ssh_user: str = "ubuntu") -> Dict[str, Any]:
    """Get GPU information from nvidia-smi on a specific node or locally."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
               "--format=csv,nounits,noheader"]

        if node_ip and node_ip != "local":
            # Run on remote node via SSH
            cmd = ["ssh", f"{ssh_user}@{node_ip}"] + cmd

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            return {"error": f"nvidia-smi failed: {result.stderr}"}

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "utilization_percent": int(parts[4]),
                    "memory_percent": round(100 * int(parts[2]) / int(parts[3]), 1)
                })

        return {"gpus": gpus}

    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out"}
    except Exception as e:
        return {"error": str(e)}


def get_ray_cluster_status(ssh_user: str = "ubuntu") -> Dict[str, Any]:
    """Get comprehensive Ray cluster status including GPU allocations."""

    if not ray.is_initialized():
        return {"error": "Ray is not initialized"}

    # Get cluster resources
    total_resources = ray.cluster_resources()
    available_resources = ray.available_resources()

    # Get node information
    nodes = ray.nodes()
    node_info = []

    for node in nodes:
        if node['Alive']:
            node_resources = node['Resources']
            node_id = node['NodeID'][:8]  # Short ID for display

            # Extract IP address from node
            node_ip = node.get('NodeManagerAddress', 'unknown')

            node_data = {
                "id": node_id,
                "ip": node_ip,
                "cpus": {
                    "total": node_resources.get('CPU', 0),
                    "used": node_resources.get('CPU', 0) - available_resources.get(f'node:{node["NodeID"]}.CPU', 0)
                },
                "gpus": {
                    "total": node_resources.get('GPU', 0),
                    "used": node_resources.get('GPU', 0) - available_resources.get(f'node:{node["NodeID"]}.GPU', 0)
                },
                "memory_mb": node_resources.get('memory', 0) / (1024 * 1024)
            }

            # Get detailed GPU info from nvidia-smi
            if node_data["gpus"]["total"] > 0:
                smi_info = get_nvidia_smi_info(
                    node_ip if node_ip != "unknown" else "local",
                    ssh_user=ssh_user,
                )
                if "gpus" in smi_info:
                    node_data["gpu_details"] = smi_info["gpus"]

            node_info.append(node_data)

    # Get running tasks/actors (simplified - Ray doesn't expose detailed task info easily)
    # This would need custom tracking in the actual experiments

    return {
        "cluster": {
            "total_cpus": total_resources.get('CPU', 0),
            "available_cpus": available_resources.get('CPU', 0),
            "total_gpus": total_resources.get('GPU', 0),
            "available_gpus": available_resources.get('GPU', 0),
        },
        "nodes": node_info,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def format_gpu_status(status: Dict[str, Any], verbose: bool = False) -> str:
    """Format the GPU status for console output."""

    if "error" in status:
        return f"Error: {status['error']}"

    output = []
    output.append("=" * 80)
    output.append(f"RAY CLUSTER GPU STATUS - {status['timestamp']}")
    output.append("=" * 80)

    # Cluster summary
    cluster = status['cluster']
    output.append("\nCLUSTER RESOURCES:")
    output.append(f"  CPUs: {cluster['available_cpus']:.0f}/{cluster['total_cpus']:.0f} available")
    output.append(f"  GPUs: {cluster['available_gpus']:.0f}/{cluster['total_gpus']:.0f} available")

    # Node details
    output.append("\nNODE DETAILS:")
    for node in status['nodes']:
        output.append(f"\n  Node {node['id']} ({node['ip']}):")
        output.append(f"    CPUs: {node['cpus']['used']:.0f}/{node['cpus']['total']:.0f} in use")
        output.append(f"    GPUs: {node['gpus']['used']:.0f}/{node['gpus']['total']:.0f} in use")

        if "gpu_details" in node and verbose:
            output.append("    GPU Details:")
            for gpu in node['gpu_details']:
                output.append(f"      GPU {gpu['index']} ({gpu['name']}):")
                output.append(f"        Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB ({gpu['memory_percent']}%)")
                output.append(f"        Utilization: {gpu['utilization_percent']}%")

    # GPU allocation summary
    output.append("\nGPU ALLOCATION SUMMARY:")
    total_gpus = int(cluster['total_gpus'])
    used_gpus = int(cluster['total_gpus'] - cluster['available_gpus'])
    free_gpus = int(cluster['available_gpus'])

    # Visual representation of GPU allocation
    gpu_bar = ['[']
    for i in range(total_gpus):
        if i < used_gpus:
            gpu_bar.append('■')  # Used GPU
        else:
            gpu_bar.append('□')  # Free GPU
    gpu_bar.append(']')

    output.append(f"  {''.join(gpu_bar)} {used_gpus}/{total_gpus} GPUs in use")

    output.append("=" * 80)

    return '\n'.join(output)


def monitor_loop(interval: int = 5, verbose: bool = False):
    """Continuously monitor GPU status."""
    print("Starting GPU monitor (press Ctrl+C to stop)...")
    print(f"Refreshing every {interval} seconds\n")

    try:
        while True:
            # Clear screen (works on Unix-like systems)
            print("\033[2J\033[H", end="")

            status = get_ray_cluster_status()
            print(format_gpu_status(status, verbose=verbose))

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU usage in Ray cluster")
    parser.add_argument(
        "--head-node-ip",
        type=str,
        default=os.getenv("AIDE_HEAD_NODE_IP"),
        help="IP address of Ray head node (defaults to AIDE_HEAD_NODE_IP).",
    )
    parser.add_argument(
        "--ray-port",
        type=int,
        default=int(os.getenv("AIDE_RAY_PORT", "6379")),
        help="Ray GCS port (default: AIDE_RAY_PORT or 6379).",
    )
    parser.add_argument(
        "--ssh-user",
        type=str,
        default=os.getenv("AIDE_SSH_USER", "ubuntu"),
        help="SSH user for remote nvidia-smi checks (default: AIDE_SSH_USER or ubuntu).",
    )
    parser.add_argument("--once", action="store_true",
                       help="Show status once and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed GPU information")
    parser.add_argument("--interval", type=int, default=5,
                       help="Refresh interval in seconds (default: 5)")
    parser.add_argument("--json", action="store_true",
                       help="Output in JSON format")

    args = parser.parse_args()

    # Initialize Ray connection
    try:
        if args.head_node_ip and args.head_node_ip != "local":
            ray.init(address=f"{args.head_node_ip}:{args.ray_port}")
        else:
            ray.init(address="auto")
    except:
        print("Failed to connect to Ray. Trying to initialize local instance...")
        ray.init()

    # Get status
    status = get_ray_cluster_status(ssh_user=args.ssh_user)

    if args.json:
        print(json.dumps(status, indent=2))
    elif args.once:
        print(format_gpu_status(status, verbose=args.verbose))
    else:
        monitor_loop(interval=args.interval, verbose=args.verbose)

    ray.shutdown()


if __name__ == "__main__":
    main()
