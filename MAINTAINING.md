# Maintainer's Guide

This document provides instructions for Google employees to maintain the codebase.

## Google-specific Style Checking

Google uses `gpylint`, which is a wrapper around `pylint`, to check for `pylint`-related errors as well as additional Google-specific code formatting errors that `pylint` does not handle. These Google-specific errors begin with "g-" and can be ignored / disabled using the usual `pylint` pragma comments.

These `gpylint` checks are performed automatically by internal Google tools, including during a Copybara sync (see "Copybara Sync" section below). It looks like we might not be able to configure `gpylint` using the existing ".pylintrc" config file, so to minimize code diffs during the sync process, we may want to run checks manually in preparation for a sync.

To run the style checker manually:

```sh
# check all files:
gpylint smart_control --ignore=proto

# check a specific file:
gpylint smart_control/path/to/file.py
```

This may produce verbose outputs, which may be helpful for specific errors but which may be overwhelming when there are many errors. To control and reduce the format of the error messages:

```sh
gpylint smart_control --ignore=proto --msg-template="{path}:{line}: [{msg_id}({symbol})]"
```

To ignore and/or check for certain messages, using the corresponding [message code](https://goto.google.com/gpylint-faq):

```sh
# disabling certain messages:
gpylint smart_control --ignore=proto --disable=g-bad-import-order,g-bad-todo

# checking for a specific message:
gpylint smart_control --ignore=proto --disable=all --enable=C6113
```

## Copybara Sync

We are using [Copybara](https://github.com/google/copybara) to manage the code sync process between GitHub (open source) and Google (internal) codebases.

### Setup Copybara

Follow the `go/copybara-setup` instructions to setup Copybara on your Google machine.

This involves setting a bash alias for the `copybara` CLI.

If successful, these commands should resolve without error:

```sh
copybara version

copybara help
```

### Update Copybara Config

Ensure the "copy.bara.sky" file in the root directory of the repository is up to date. This file exists in the Google codebase only.

### GitHub to Google

Perform a dry run sync from GitHub to Google:

```sh
# copybara third_party/py/smart_buildings/copy.bara.sky  --init-history --dry-run --force

# or:
copybara copy.bara.sky  --init-history --dry-run
```

This creates a CL on the Google side, pulling in all the changes from GitHub to Google.

Remember to merge this CL on the Google side before proceeding!

### Google to GitHub

Invoke the “git_to_third_party” workflow in the .sky file.

This will overwrite the GitHub repository, so make sure to merge the changes from GitHub to Google first!
