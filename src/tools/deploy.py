"""Build the site and copy it + the source into the GitHub working copy (/home/claude/socialp).
- /home/claude/socialp/index.html, stats.html = deploy head + dist/<same name> + closing tags
- /home/claude/socialp/src/         = this source tree (without dist/ and caches)
Does not commit or push."""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "/home/claude/socialp"
subprocess.run([sys.executable, os.path.join(ROOT, "tools", "merge14.py")], check=True, cwd=ROOT)
subprocess.run([sys.executable, os.path.join(ROOT, "build.py")], check=True, cwd=ROOT)
head = open(os.path.join(ROOT, "tools", "deploy_head.html"), encoding="utf-8").read()
for name in ("index.html", "stats.html"):   # home page + one page per course
    body = open(os.path.join(ROOT, "dist", name), encoding="utf-8").read()
    open(os.path.join(REPO, name), "w", encoding="utf-8").write(head + body + "\n</body></html>\n")
import shutil
dst = os.path.join(REPO, "src")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("dist", "__pycache__", "*.pyc"))
print("deployed to", REPO)
