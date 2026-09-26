# Browser tools

Enable browser interaction explicitly:

```bash
noah browser setup
```

This saves a Playwright MCP entry in `~/.config/noah-code/mcp.json`. It does not
launch a browser or install software during setup. Restart Noah to attach the
configured server through normal MCP startup. Setup preserves other servers and
refuses to replace an existing entry named `browser`.

Install Noah's MCP extra and have Node.js 18 or newer with `npx` available. From
a source checkout, install the extra with `uv sync --extra mcp`. On first
connection, `npx` may download the pinned `@playwright/mcp@0.0.82` package.

The preset uses an isolated in-memory profile and runs headlessly. It does not
connect to existing browser tabs or reuse your regular login profile. Browser
storage disappears when its session closes. Choose a visible window with
`noah browser setup --headed`; select a browser with `--browser firefox`,
`--browser webkit`, or `--browser msedge` (default: `chrome`). The selected browser
must be installed; Playwright MCP exposes `browser_install` when installation
is needed.

Ask Noah to open your local application, inspect its page snapshot, exercise a
form, inspect console/network output, or capture a screenshot. The tools appear
under `self.browser` after attachment. Browser mutations, including navigation
and script evaluation, use MCP permission checks and are denied in plan mode.
Unknown browser actions are treated as mutations. Snapshot, console, and network
inspection are read-only unless an output filename is supplied. Interrupted
browser actions are not automatically replayed; this is not transactional
rollback of changes made in the browser or on a website.

To disable the server, add `"disabled": true` to its entry in `mcp.json` and
restart Noah. Existing configuration and permission precedence still apply.

The preset follows Microsoft's [Playwright MCP configuration](https://github.com/microsoft/playwright-mcp#configuration)
and [isolated-profile behavior](https://github.com/microsoft/playwright-mcp#user-profile).
Package version and Node requirement were checked against the
[official npm package metadata](https://registry.npmjs.org/@playwright/mcp/0.0.82)
on September 26, 2026. MCP connection tests use a local fake server; browser
launch and npm installation require a configured local environment.
