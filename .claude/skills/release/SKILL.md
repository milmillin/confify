---
name: release
description: Create a GitHub release that triggers PyPI publishing
disable-model-invocation: true
allowed-tools: Bash(git *), Bash(gh *), Bash(pytest *), Bash(bash test-static-type.sh*)
---

Create a GitHub release for the confify package. This triggers the publish-to-PyPI workflow.

## Steps

1. Read the current version from `confify/__init__.py` (`__version__`).
2. Run `git status` to ensure the working tree is clean. If not, stop and tell the user.
3. Run `pytest` to ensure all tests pass. If any fail, stop and tell the user.
4. Run `bash test-static-type.sh` to ensure all static type checks pass. If any fail, stop and tell the user.
5. Run `git log` to collect all commits since the last git tag (or all commits if no tag exists). Summarize them into release notes.
6. Confirm with the user before proceeding, showing:
   - The version tag that will be created (e.g. `v0.0.15`)
   - The generated release notes
7. After user approval:
   - Create and push an annotated git tag: `git tag -a v<version> -m "v<version>"`
   - Push the tag: `git push origin v<version>`
   - Create a GitHub release via `gh release create v<version> --title "v<version>" --notes "<release notes>"`
8. Print the release URL when done.
