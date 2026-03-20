# Git Workflow Notes

## Branching Strategy

Use short-lived feature branches off `main`. Name them descriptively: `feature/add-auth`, `fix/login-redirect`, `chore/update-deps`.

## Commit Messages

Follow conventional commits: `type(scope): description`

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `chore:` — maintenance tasks
- `refactor:` — code restructuring without behaviour change

## Pull Requests

- Keep PRs small and focused on a single change
- Write a clear description of what and why
- Request reviews from relevant team members
- Squash merge to keep history clean

## Useful Commands

- `git stash` — temporarily shelve uncommitted changes
- `git rebase -i HEAD~3` — interactive rebase to clean up recent commits
- `git bisect` — binary search to find the commit that introduced a bug
- `git log --oneline --graph` — visualise branch history
