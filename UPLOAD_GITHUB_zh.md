# 上传到 King-play/EvoGen-Harness

## 本次应该替换什么

将 ZIP 解压后的 `EvoGen-Harness/` **里面的内容**合并到仓库根目录。新版 `README.md` 替换原来的网页介绍 README；源码、`assets/readme/`、`documentation/`、`data/evolution/`、引用文件和隐藏配置要一起上传。

**保留现有 `docs/` 文件夹，不删除、不覆盖。** 这份代码包没有携带该目录，不会替你回退已修改的 arXiv 链接、封面和作者信息。新代码文档使用 `documentation/`，不是 `docs/`。

## 推荐：本地 Git 或 GitHub Desktop

先克隆当前远端仓库（或在自己的已有克隆中拉取最新内容），确保没有未提交的个人改动：

```bash
git clone https://github.com/King-play/EvoGen-Harness.git
cd EvoGen-Harness
git status --short
git switch -c release/code-and-readme
```

然后用文件管理器将解压目录中的内容复制合并到这个仓库。显示隐藏文件，确保 `.github/`、`.gitignore`、`.env.example` 没有漏掉；保留现有其他文件。不要使用带 `--delete` 的同步方式。

上传前检查：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q -rs
python scripts/check_release.py
git status --short
git diff --name-only -- docs
git diff -- README.md
```

`git diff --name-only -- docs` 应没有输出。确认没有个人运行产物或密钥后，暂存本次代码包对应的路径，避免顺手提交其他本地文件：

```bash
git add README.md README_zh-CN.md CITATION.bib CITATION.cff LICENSE \
  CONTRIBUTING.md SECURITY.md THIRD_PARTY_NOTICES.md UPLOAD_GITHUB_zh.md \
  pyproject.toml requirements.txt requirements-dev.txt .gitignore .env.example .github \
  gen_harness configs examples data scripts tests assets/readme documentation artifacts licenses
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only -- docs
```

确认暂存区没有 `docs/` 修改，再提交推送：

```bash
git commit -m "Release EvoGen-Harness implementation and research README"
git push -u origin release/code-and-readme
```

在 GitHub 打开 Pull Request，检查 Files changed、图片显示和自动检查，再合并到默认分支。本流程不需要强制推送，也不需要删除原仓库。

## 使用网页上传

GitHub 网页每批最多上传 100 个文件，本包超过这一数量，需要按目录分批。进入仓库根目录上传**解压后的实际文件**，不是上传 ZIP，也不要多套一层 `EvoGen-Harness/`。网页上传不便处理隐藏文件时，改用 GitHub Desktop/本地 Git。

官方说明：[上传文件](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)。

## 发布后检查

查看仓库首页 Overview、图片、中文入口、Paper/Project 链接和引用是否正确；确认 Actions 中的代码检查结果。既有 GitHub Pages 的 `main /docs` 设置保持不变。新的 CI 只测试代码，不会代替你运行 GPU benchmark，也没有自动上传实验产物。

以后编辑主 README 时，用 `python scripts/check_release.py` 检查相对链接、引用一致性和数据完整性。个人 API Key 与模型路径只放本地环境文件；不要通过更新公开示例文件来填入真实密钥。
