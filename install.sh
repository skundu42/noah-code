#!/bin/sh

set -eu

UV_INSTALL_URL="${UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
NOAH_CODE_PACKAGE="${NOAH_CODE_PACKAGE:-noah-code[mcp,tracing]}"

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'noah-code installer: %s\n' "$*" >&2
    exit 1
}

case "$(uname -s 2>/dev/null || true)" in
    Darwin|Linux) ;;
    *) fail "only macOS and Linux are currently supported" ;;
esac

case "$(uname -m 2>/dev/null || true)" in
    arm64|aarch64|x86_64|amd64) ;;
    *) fail "unsupported CPU architecture: $(uname -m 2>/dev/null || printf unknown)" ;;
esac

temporary_dir="$(mktemp -d 2>/dev/null || mktemp -d -t noah-code)"
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM

download() {
    source_url="$1"
    destination="$2"
    if command -v curl >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -LsSf "$source_url" -o "$destination"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$destination" "$source_url"
    else
        fail "curl or wget is required to download the installer"
    fi
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    return 1
}

if uv_path="$(find_uv)"; then
    say "Using uv at $uv_path"
else
    say "Installing the self-contained uv package manager..."
    download "$UV_INSTALL_URL" "$temporary_dir/uv-installer.sh"
    sh "$temporary_dir/uv-installer.sh"
    uv_path="$(find_uv)" || fail "uv installed but could not be located"
fi

package="$NOAH_CODE_PACKAGE"
if [ -n "${NOAH_CODE_VERSION:-}" ]; then
    package="${NOAH_CODE_PACKAGE}==${NOAH_CODE_VERSION}"
fi

say "Installing $package with a managed Python 3.12 runtime..."
"$uv_path" tool install --managed-python --python 3.12 --no-build --force "$package"
"$uv_path" tool update-shell >/dev/null 2>&1 || true

say ""
say "Noah Code is installed. Open a new terminal, then run:"
say ""
say "  noah ."
say ""
say "Updates are installed automatically. Run 'noah update' to update immediately."
