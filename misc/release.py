import argparse
import subprocess
from dataclasses import dataclass
from typing import Literal


@dataclass
class VersionInfo:
    version: str
    bump_type: Literal["patch", "minor", "major"]


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def get_new_version(bump_type: Literal["patch", "minor", "major"]) -> VersionInfo:
    result = run_command(cmd=["poetry", "version", bump_type])
    version = result.stdout.strip().split()[-1]
    return VersionInfo(version=version, bump_type=bump_type)


def git_commands(version: str) -> None:
    commands = [
        [
            "git",
            "add",
            "pyproject.toml",
            "uv.lock",
        ],
        ["git", "commit", "-m", f"Bump version to {version}"],
        ["git", "tag", version],
    ]

    push_command = ["git", "push"]
    commands.append(push_command)

    commands.append(["git", "push", "origin", "--tags"])

    for cmd in commands:
        run_command(cmd=cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate release workflow")
    parser.add_argument(
        "bump_type",
        choices=["patch", "minor", "major"],
        help="Version bump type",
    )

    args = parser.parse_args()

    try:
        version_info = get_new_version(bump_type=args.bump_type)
        print(f"Bumping version to: {version_info.version}")

        git_commands(version=version_info.version)
        print("Successfully completed all git commands")

        print("\nRelease workflow completed successfully!")
        print("Don't forget to update Github release notes!")

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        print(f"Output: {e.output}")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
