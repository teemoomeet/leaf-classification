"""在无法直连 github.com 的网络环境下，改用 GitHub REST API 推送本地仓库。

适用场景
--------
部分网络（校园网 / 公司出口）会屏蔽 ``github.com:443``，导致

* GitHub 网页打不开，
* ``git push`` 直接超时，

但 ``api.github.com`` 依然可用。此脚本就利用这一点：把本地 git 历史逐条
"重放"到远端仓库，产出与 ``git push`` 等价的提交历史（SHA 会不同，因为
GitHub 服务端会重新计算，但每个提交的内容、作者、时间、提交信息都保留）。

用法
----
1. 在任意能访问 GitHub 的设备上生成 Personal Access Token（勾选 ``repo`` 权限）::

       https://github.com/settings/tokens/new

2. 设置环境变量并运行::

       # Windows PowerShell
       $env:GITHUB_TOKEN = "ghp_xxxxxxxx"
       python scripts/push_via_api.py --repo flavia-leaf-recognition --public

       # Windows CMD
       set GITHUB_TOKEN=ghp_xxxxxxxx
       python scripts/push_via_api.py --repo flavia-leaf-recognition --public

   仓库属主默认取 token 所属账号，可用 ``--owner`` 覆盖（例如推到组织下）。

3. 推送完成后，把远程地址配置到本地（github.com 不通时该命令会失败，不影响
   远端内容，等网络恢复后再执行即可）::

       git remote add origin https://github.com/<owner>/<repo>.git

依赖：仅 Python 标准库。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
UA = "leaves-api-pusher/1.0"


# --------------------------------------------------------------------------- #
# HTTP 封装
# --------------------------------------------------------------------------- #
def _call(method: str, path: str, token: str, payload: dict | None = None,
          ok: tuple[int, ...] = (200, 201)) -> dict:
    """调用 GitHub API，返回解析后的 JSON。"""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": UA,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            code = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if exc.code in ok:
            return json.loads(body) if body.strip() else {}
        raise RuntimeError(f"{method} {path} 失败 [{exc.code}]: {body[:400]}") from exc
    if code not in ok:
        raise RuntimeError(f"{method} {path} 返回 {code}: {body[:400]}")
    return json.loads(body) if body.strip() else {}


def _git(*args: str) -> str:
    """执行 git 命令并返回 stdout（按原始字节解码，避免中文提交信息乱码）。"""
    out = subprocess.run(["git", *args], capture_output=True, check=True)
    return out.stdout.decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# 读取本地仓库
# --------------------------------------------------------------------------- #
def local_commits() -> list[str]:
    """按从旧到新的顺序返回 HEAD 可达的全部提交。"""
    shas = _git("rev-list", "--reverse", "HEAD").split()
    if not shas:
        raise RuntimeError("本地仓库还没有任何提交")
    return shas


def commit_meta(sha: str) -> dict:
    """读取单个提交的元信息。"""
    out = _git("log", "-1", "--format=%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%P%x00%B", sha)
    parts = out.split("\x00", 7)
    author_name, author_email, author_date = parts[0], parts[1], parts[2]
    committer_name, committer_email, committer_date = parts[3], parts[4], parts[5]
    parents = parts[6].split()
    message = parts[7].rstrip("\n")
    return {
        "message": message,
        "author": {"name": author_name, "email": author_email, "date": author_date},
        "committer": {"name": committer_name, "email": committer_email, "date": committer_date},
        "parents": parents,
    }


def tree_entries(sha: str) -> list[dict]:
    """列出该提交的完整文件树（扁平结构，路径为相对仓库根的完整路径）。"""
    raw = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--full-tree", sha],
        capture_output=True, check=True,
    ).stdout.decode("utf-8", "replace")
    entries: list[dict] = []
    for item in raw.split("\x00"):
        if not item:
            continue
        meta, path = item.split("\t", 1)
        mode, otype, osha = meta.split()
        if otype == "blob":
            entries.append({"path": path, "mode": mode, "type": "blob", "sha": osha})
        elif otype == "commit":  # 子模块
            entries.append({"path": path, "mode": "160000", "type": "commit", "sha": osha})
    return entries


def blob_bytes(sha: str) -> bytes:
    """读取 blob 原始内容。"""
    return subprocess.run(["git", "cat-file", "blob", sha], capture_output=True, check=True).stdout


# --------------------------------------------------------------------------- #
# 推送到 GitHub
# --------------------------------------------------------------------------- #
def ensure_repo(token: str, owner: str | None, repo: str, *, private: bool,
                description: str) -> tuple[str, str]:
    """确保仓库存在，返回 (owner, repo)。"""
    if owner is None:
        owner = _call("GET", "/user", token)["login"]
        print(f"  token 所属账号：{owner}")

    try:
        info = _call("GET", f"/repos/{owner}/{repo}", token)
        print(f"  仓库已存在：{info['full_name']}")
        return owner, repo
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise

    print(f"  创建仓库 {owner}/{repo} ...")
    # 组织下的仓库要走 /orgs/{org}/repos
    me = _call("GET", "/user", token)["login"]
    path = "/user/repos" if owner == me else f"/orgs/{owner}/repos"
    info = _call("POST", path, token, {
        "name": repo,
        "description": description,
        "private": private,
        "has_issues": True,
        "has_wiki": False,
        "auto_init": False,
    })
    print(f"  创建成功：{info['html_url']}")
    return owner, repo


def push_history(token: str, owner: str, repo: str) -> tuple[int, list[str]]:
    """把本地提交逐条重放到远端，返回 (提交数, 远端提交 SHA 列表)。"""
    blob_cache: dict[str, str] = {}      # 本地 blob sha -> 远端 blob sha
    remote_parent: str | None = None     # 上一条远端提交
    pushed: list[str] = []

    shas = local_commits()
    print(f"  待推送提交 {len(shas)} 条")

    for idx, sha in enumerate(shas, 1):
        meta = commit_meta(sha)
        subject = meta["message"].splitlines()[0] if meta["message"] else "(无提交信息)"
        print(f"  [{idx}/{len(shas)}] {sha[:8]} {subject[:56]}")

        # 1) 上传本提交涉及的所有 blob
        entries = tree_entries(sha)
        remote_tree: list[dict] = []
        uploaded = 0
        for ent in entries:
            if ent["type"] != "blob":
                remote_tree.append(ent)
                continue
            rsha = blob_cache.get(ent["sha"])
            if rsha is None:
                content = base64.b64encode(blob_bytes(ent["sha"])).decode("ascii")
                rsha = _call("POST", f"/repos/{owner}/{repo}/git/blobs", token,
                             {"content": content, "encoding": "base64"})["sha"]
                blob_cache[ent["sha"]] = rsha
                uploaded += 1
            remote_tree.append({"path": ent["path"], "mode": ent["mode"],
                                "type": "blob", "sha": rsha})
        print(f"        文件 {len(entries)} 个（新上传 {uploaded}，复用缓存 {len(entries) - uploaded}）")

        # 2) 建树
        tree_sha = _call("POST", f"/repos/{owner}/{repo}/git/trees", token,
                         {"tree": remote_tree})["sha"]

        # 3) 建提交（parent 用远端上一条，保证远端历史线性且连续）
        payload = {
            "message": meta["message"],
            "tree": tree_sha,
            "author": meta["author"],
            "committer": meta["committer"],
        }
        if remote_parent:
            payload["parents"] = [remote_parent]
        commit = _call("POST", f"/repos/{owner}/{repo}/git/commits", token, payload)
        remote_parent = commit["sha"]
        pushed.append(remote_parent)

    return len(pushed), pushed


def update_branch(token: str, owner: str, repo: str, sha: str, branch: str) -> None:
    """把分支指向最新提交（不存在则创建）。"""
    try:
        _call("PATCH", f"/repos/{owner}/{repo}/git/refs/heads/{branch}", token,
              {"sha": sha, "force": True})
        print(f"  分支 {branch} 已更新")
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise
        _call("POST", f"/repos/{owner}/{repo}/git/refs", token,
              {"ref": f"refs/heads/{branch}", "sha": sha})
        print(f"  分支 {branch} 已创建")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="通过 GitHub API 推送本地仓库（适用于 github.com 被屏蔽的网络）")
    parser.add_argument("--owner", default=None, help="仓库属主，默认取 token 所属账号")
    parser.add_argument("--repo", required=True, help="仓库名")
    parser.add_argument("--branch", default="main", help="目标分支，默认 main")
    parser.add_argument("--private", action="store_true", help="创建私有仓库")
    parser.add_argument("--description", default="", help="仓库描述")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("错误：请先设置环境变量 GITHUB_TOKEN（GitHub Personal Access Token）",
              file=sys.stderr)
        return 2

    print("[1/4] 校验 token")
    account = _call("GET", "/user", token)
    print(f"  已认证：{account['login']}")

    print("[2/4] 准备远端仓库")
    owner, repo = ensure_repo(token, args.owner, args.repo,
                              private=args.private, description=args.description)

    print("[3/4] 上传提交历史")
    count, shas = push_history(token, owner, repo)
    if not shas:
        print("  没有任何提交可推送", file=sys.stderr)
        return 1
    update_branch(token, owner, repo, shas[-1], args.branch)

    print("[4/4] 完成")
    url = f"https://github.com/{owner}/{repo}"
    print(f"  仓库地址：{url}")
    print(f"  已推送 {count} 条提交，{len(shas)} 个版本")
    print(f"  本地配置远程：git remote add origin {url}.git")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
