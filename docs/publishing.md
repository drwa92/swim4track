# Publish the repository

Extract the ZIP and use the `swim4track/` folder as the repository root. The
archive includes the Python package, ROS adapter, original checkpoint, docs,
tests, GitHub Actions workflow and the short real demo. Build products, virtual
environments, bags and training outputs are excluded.

Review [release_checklist.md](release_checklist.md), then create an empty GitHub
repository in the intended account. Copy its HTTPS or SSH URL. From the extracted
`swim4track/` folder, use your configured Git identity:

```bash
git init -b main
git add .
git commit -m "Initial Swim4Track release"
read -r -p "Paste the empty GitHub repository URL: " repository_url
git remote add origin "$repository_url"
git push -u origin main
```

These commands are for a new repository. If the directory already has Git
history or an `origin`, inspect it before changing remotes or branches. GitHub
credentials remain in your normal Git authentication flow; do not put them in
source files or commit messages.

Once the URL exists, add it to `CITATION.cff`. Let the included CI complete and
inspect its logs. The README video poster links to the bundled MP4; GitHub can
preview the video file. The original large recording is intentionally not in
the repository. The archive itself is not evidence of a successful public push.
