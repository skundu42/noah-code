/* ==========================================================================
   Noah Code Landing Page — Zero-Dependency Interactive Cockpit Controller
   ========================================================================== */

(function () {
  "use strict";

  const THEME_META_COLORS = {
    "atom-one-dark": "#0b0d12",
    "noah-ocean": "#07151d",
    "daylight": "#f7f7f9",
  };

  /* ------------------------------------------------------------------------
     1. Authentic Data Catalogs (from src/noah_code/)
     ------------------------------------------------------------------------ */

  const SUBAGENT_TRACES = {
    explore: {
      title: "explore · read-only · completed (1.1s)",
      output:
        "✓ Read src/api/search.py, src/db/queries.py, src/schemas/pagination.py (+1) · Condensed report (620 chars): `/v1/search` sorts by `(created_at DESC, id DESC)`. Keyset cursor `(ts_ms, ulid)` avoids OFFSET scan and preserves existing `page` fallback.",
    },
    planner: {
      title: "planner · read-only · synthesizing (2.4s)",
      output:
        "✓ Inspected tie-breaker index `idx_events_created_id` · Verified base64url HMAC cursor payload rejects tampered offsets with HTTP 400 and keeps OpenAPI schema backward-compatible.",
    },
    "builder-api": {
      title: "general [worktree: .noah-worktrees/task-api-7c2] · isolated writer (3.8s)",
      output:
        "⬡ Linked Git worktree branched from HEAD · Applied exact `Match` edit in `src/api/search.py` (+42 -9) and created `src/schemas/cursor.py` (+31) · pyright + ruff: 0 diagnostics.",
    },
    "builder-tests": {
      title: "general [worktree: .noah-worktrees/task-tests-9d4] · isolated writer (3.6s)",
      output:
        "⬡ Linked Git worktree branched from HEAD · Added 8 boundary & tie-breaker tests in `tests/test_search.py` (+68) · `pytest -q tests/test_search.py` -> 18 passed in 0.48s.",
    },
  };

  const DIFF_FILES = {
    "src/api/search.py": {
      hunkMeta: "Hunk 1 of 2 · @@ -44,8 +44,12 @@",
      staged: false,
      reverted: false,
      rows: [
        {
          type: "ctx",
          oldLn: "44",
          newLn: "44",
          oldHtml: "async def list_search_results(db: AsyncSession, q: str,",
          newHtml: "async def list_search_results(db: AsyncSession, q: str,",
        },
        {
          type: "mod",
          oldLn: "45",
          newLn: "45",
          oldHtml: '-     <mark class="intra-del">page: int = 1,</mark> limit: int = <mark class="intra-del">25</mark>) -&gt; SearchPage:',
          newHtml: '+     <mark class="intra-add">cursor: str | None = None, page: int = 1,</mark> limit: int = <mark class="intra-add">50</mark>) -&gt; SearchPage:',
        },
        {
          type: "mod",
          oldLn: "46",
          newLn: "46",
          oldHtml: '-     stmt = build_search_stmt(q).<mark class="intra-del">offset((page - 1) * limit)</mark>.limit(limit)',
          newHtml: '+     token = <mark class="intra-add">decode_cursor(cursor) if cursor else None</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "47",
          oldHtml: "",
          newHtml: '+     stmt = build_search_stmt(q, <mark class="intra-add">after=token, page=page</mark>).limit(<mark class="intra-add">limit + 1</mark>)',
        },
        {
          type: "ctx",
          oldLn: "47",
          newLn: "48",
          oldHtml: "    rows = (await db.execute(stmt)).scalars().all()",
          newHtml: "    rows = (await db.execute(stmt)).scalars().all()",
        },
        {
          type: "mod",
          oldLn: "48",
          newLn: "49",
          oldHtml: '-     return SearchPage(items=<mark class="intra-del">rows</mark>, next_cursor=<mark class="intra-del">None</mark>)',
          newHtml: '+     has_more = <mark class="intra-add">len(rows) &gt; limit</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "50",
          oldHtml: "",
          newHtml: '+     items = <mark class="intra-add">rows[:limit]</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "51",
          oldHtml: "",
          newHtml: '+     return SearchPage(items=<mark class="intra-add">items</mark>, next_cursor=<mark class="intra-add">encode_cursor(items[-1]) if has_more else None</mark>)',
        },
      ],
    },
    "src/schemas/cursor.py": {
      hunkMeta: "Hunk 1 of 1 · @@ -0,0 +1,14 @@",
      staged: true,
      reverted: false,
      rows: [
        {
          type: "add",
          oldLn: "",
          newLn: "1",
          oldHtml: "",
          newHtml: '+ <mark class="intra-add">from dataclasses import dataclass</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "2",
          oldHtml: "",
          newHtml: '+ <mark class="intra-add">import base64, hashlib, hmac, json</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "4",
          oldHtml: "",
          newHtml: '+ <mark class="intra-add">@dataclass(frozen=True, slots=True)</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "5",
          oldHtml: "",
          newHtml: '+ <mark class="intra-add">class KeysetCursor:</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "6",
          oldHtml: "",
          newHtml: '+     created_at_ms: <mark class="intra-add">int</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "7",
          oldHtml: "",
          newHtml: '+     record_id: <mark class="intra-add">str</mark>',
        },
      ],
    },
    "tests/test_search.py": {
      hunkMeta: "Hunk 1 of 1 · @@ -88,4 +88,11 @@",
      staged: false,
      reverted: false,
      rows: [
        {
          type: "ctx",
          oldLn: "88",
          newLn: "88",
          oldHtml: "async def test_search_pagination_tie_breaker(client):",
          newHtml: "async def test_search_pagination_tie_breaker(client):",
        },
        {
          type: "mod",
          oldLn: "89",
          newLn: "89",
          oldHtml: '-     res = await client.get(<mark class="intra-del">"/v1/search?q=invoice&amp;page=2"</mark>)',
          newHtml: '+     first = await client.get(<mark class="intra-add">"/v1/search?q=invoice&amp;limit=2"</mark>)',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "90",
          oldHtml: "",
          newHtml: '+     nxt = <mark class="intra-add">first.json()["next_cursor"]</mark>',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "91",
          oldHtml: "",
          newHtml: '+     second = await client.get(<mark class="intra-add">f"/v1/search?q=invoice&amp;cursor={nxt}"</mark>)',
        },
        {
          type: "add",
          oldLn: "",
          newLn: "92",
          oldHtml: "",
          newHtml: '+     assert <mark class="intra-add">set(first.ids()).isdisjoint(second.ids())</mark>',
        },
      ],
    },
  };

  const MODEL_CATALOG = [
    {
      id: "anthropic/claude-sonnet-5.5",
      shortName: "claude-sonnet-5.5",
      ctx: "200k ctx",
      family: "anthropic",
      desc: "◆ Coding flagship · Anthropic Claude Sonnet 5.5",
    },
    {
      id: "anthropic/claude-opus-5.5",
      shortName: "claude-opus-5.5",
      ctx: "200k ctx",
      family: "anthropic",
      desc: "◆ Coding & architecture · Anthropic Claude Opus 5.5",
    },
    {
      id: "openai/gpt-5.3-codex",
      shortName: "gpt-5.3-codex",
      ctx: "400k ctx",
      family: "openai",
      desc: "◆ Agentic coding · OpenAI GPT-5.3 Codex (OAuth or API key)",
    },
    {
      id: "openrouter/auto-code",
      shortName: "openrouter/auto-code",
      ctx: "2M ctx",
      family: "openrouter",
      desc: "◆ Coding router · auto-selects top coding model",
    },
    {
      id: "openrouter/openai/gpt-6.1-terra-pro",
      shortName: "gpt-6.1-terra-pro",
      ctx: "1.05M ctx",
      family: "openrouter",
      desc: "◆ OpenRouter openai/* · Frontier reasoning & code",
    },
    {
      id: "openrouter/anthropic/claude-sonnet-5.5",
      shortName: "or/claude-sonnet-5.5",
      ctx: "200k ctx",
      family: "openrouter",
      desc: "◆ OpenRouter anthropic/* · Zero-typing curated route",
    },
    {
      id: "qwen/qwen3-coder-next",
      shortName: "qwen3-coder-next",
      ctx: "262k ctx",
      family: "frontier",
      desc: "◆ Open-weight coding · Qwen3 Coder Next",
    },
    {
      id: "deepseek/deepseek-v4.1-flash",
      shortName: "deepseek-v4.1-flash",
      ctx: "1M ctx",
      family: "frontier",
      desc: "◆ Fast high-value coding · DeepSeek V4.1 Flash",
    },
    {
      id: "google/gemini-4-hydrogen",
      shortName: "gemini-4-hydrogen",
      ctx: "1M ctx",
      family: "frontier",
      desc: "◆ 1M-ctx coding · Google Gemini 4 Hydrogen",
    },
    {
      id: "x-ai/grok-build-0.1",
      shortName: "grok-build-0.1",
      ctx: "256k ctx",
      family: "frontier",
      desc: "◆ Agentic engineering · xAI Grok Build 0.1",
    },
    {
      id: "mistralai/devstral-2512",
      shortName: "devstral-2512",
      ctx: "262k ctx",
      family: "frontier",
      desc: "◆ SWE agent model · Mistral Devstral 2512",
    },
    {
      id: "moonshotai/kimi-k2.7-code",
      shortName: "kimi-k2.7-code",
      ctx: "262k ctx",
      family: "frontier",
      desc: "◆ Specialized coding · Moonshot Kimi K2.7 Code",
    },
    {
      id: "z-ai/glm-5.3",
      shortName: "glm-5.3",
      ctx: "1M ctx",
      family: "frontier",
      desc: "◆ Agentic coding · Z.ai GLM-5.3",
    },
    {
      id: "openai/gpt-5.2-codex",
      shortName: "gpt-5.2-codex",
      ctx: "400k ctx",
      family: "openai",
      desc: "◆ Agentic coding · OpenAI GPT-5.2 Codex",
    },
  ];

  const RESUME_DETAILS = {
    ses_91f4:
      "Selected `ses_91f4` (feat/cursor-search) · Restores 14 turns, 2 isolated worktrees, todos & 92% cache prefix",
    ses_88c2:
      "Selected `ses_88c2` (fix/webhook-hmac) · Restores `$portable-review` audit transcript & checkpoint ◆ cp_81b04",
    ses_7e10:
      "Selected `ses_7e10` (main) · Restores `/team investigate` report, `.noah-code/plan.md` & usage counters",
  };

  const SHORTCUTS_AND_COMMANDS = [
    {
      key: "Ctrl+L",
      scope: "Global Picker",
      cat: "navigation",
      desc: "Open the zero-typing model picker across 80+ curated coding models & OpenRouter routes.",
    },
    {
      key: "Ctrl+D",
      scope: "Diff Review",
      cat: "review",
      desc: "Open intraline & split diff review (`G` stage hunk, `H` revert hunk, `S` split, `V` LSP diag).",
    },
    {
      key: "Ctrl+P",
      scope: "Global Picker",
      cat: "navigation",
      desc: "Search and execute built-in, user, or `.claude/commands` slash commands instantly.",
    },
    {
      key: "Ctrl+F",
      scope: "Transcript",
      cat: "composer",
      desc: "Find and jump between matching text occurrences inside the live session transcript.",
    },
    {
      key: "Ctrl+E",
      scope: "Composer",
      cat: "composer",
      desc: "Open and edit the current multi-line draft prompt in `$VISUAL` or `$EDITOR`.",
    },
    {
      key: "Alt+I / F4",
      scope: "Operator Ledger",
      cat: "review",
      desc: "Inspect live subagents, linked Git worktrees (`diff`, `apply`, `stop`), jobs, and terminals.",
    },
    {
      key: "Alt+C",
      scope: "Transcript",
      cat: "composer",
      desc: "Copy the latest fenced code block cleanly to the system clipboard with zero padding.",
    },
    {
      key: "↑ / ↓",
      scope: "Composer & Menus",
      cat: "composer",
      desc: "Navigate inline `/` command completions, `$skill` suggestions, or prior prompt history.",
    },
    {
      key: "Ctrl+B",
      scope: "Mode Switch",
      cat: "navigation",
      desc: "Toggle between implementation `build` mode and strictly read-only `plan` mode.",
    },
    {
      key: "Ctrl+G",
      scope: "Skills Picker",
      cat: "navigation",
      desc: "Search compatible `.claude/skills`, `.agents/skills`, and `.noah-code/skills` (`SKILL.md`).",
    },
    {
      key: "F9",
      scope: "Agent Teams",
      cat: "review",
      desc: "Choose a coordinated `build`, `review`, or `investigate` multi-agent workflow.",
    },
    {
      key: "Ctrl+R / Alt+Z",
      scope: "Draft Safety",
      cat: "composer",
      desc: "Recall prior session prompts while automatically saving your displaced draft for `Alt+Z`.",
    },
    {
      key: "/resume",
      scope: "Slash Command",
      cat: "slash",
      desc: "Pick and reopen your latest repository-scoped SQLite chats with full message previews.",
    },
    {
      key: "/team",
      scope: "Slash Command",
      cat: "slash",
      desc: "Launch `/team build|review|investigate OBJECTIVE` with bounded phase handoffs.",
    },
    {
      key: "/mcp",
      scope: "Slash Command",
      cat: "slash",
      desc: "Attach 12 built-in 1-click MCP presets (Context7, Playwright, GitHub…) or import `.mcp.json`.",
    },
    {
      key: "$skill",
      scope: "Inline Trigger",
      cat: "slash",
      desc: "Type `$` in the composer (e.g. `$portable-review`) to invoke a portable `SKILL.md` workflow.",
    },
    {
      key: "@file",
      scope: "Inline Trigger",
      cat: "slash",
      desc: "Attach workspace files or images inline (queued `@path` items resolve cleanly on turn start).",
    },
    {
      key: "/tokens",
      scope: "Slash Command",
      cat: "slash",
      desc: "Inspect route-aware prompt-cache hit rate, prefix stability, model wait, and session cost.",
    },
  ];

  /* ------------------------------------------------------------------------
     2. Theme Switching (Atom One Dark, Noah Ocean, Daylight)
     ------------------------------------------------------------------------ */

  function applyTheme(themeName, persist) {
    if (!THEME_META_COLORS[themeName]) return;
    document.documentElement.setAttribute("data-theme", themeName);
    const metaTheme = document.getElementById("meta-theme-color");
    if (metaTheme) {
      metaTheme.setAttribute("content", THEME_META_COLORS[themeName]);
    }
    document.querySelectorAll("[data-theme-target]").forEach(function (btn) {
      const active = btn.getAttribute("data-theme-target") === themeName;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-pressed", active ? "true" : "false");
    });
    if (persist) {
      try {
        localStorage.setItem("noah-code-theme", themeName);
      } catch (_err) {
        /* ignore storage restrictions */
      }
    }
  }

  function initTheme() {
    let saved = null;
    try {
      saved = localStorage.getItem("noah-code-theme");
    } catch (_err) {
      saved = null;
    }
    if (saved && THEME_META_COLORS[saved]) {
      applyTheme(saved, false);
    }
    document.querySelectorAll("[data-theme-target]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const target = btn.getAttribute("data-theme-target");
        if (target) applyTheme(target, true);
      });
    });
  }

  /* ------------------------------------------------------------------------
     3. Mobile Navigation Drawer
     ------------------------------------------------------------------------ */

  function initMobileNav() {
    const btn = document.getElementById("mobile-menu-btn");
    const drawer = document.getElementById("mobile-drawer");
    if (!btn || !drawer) return;

    function closeDrawer() {
      drawer.hidden = true;
      btn.setAttribute("aria-expanded", "false");
    }

    btn.addEventListener("click", function () {
      const willOpen = drawer.hidden;
      drawer.hidden = !willOpen;
      btn.setAttribute("aria-expanded", willOpen ? "true" : "false");
    });

    drawer.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", closeDrawer);
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !drawer.hidden) {
        closeDrawer();
        btn.focus();
      }
    });
  }

  /* ------------------------------------------------------------------------
     4. Install Switcher & Clipboard Copy Feedback
     ------------------------------------------------------------------------ */

  function copyToClipboard(text, buttonEl) {
    const announcer = document.getElementById("clipboard-announcer");
    const labelEl = buttonEl.querySelector(".copy-btn-label");

    function showSuccess() {
      buttonEl.classList.add("is-copied");
      if (labelEl) labelEl.textContent = "Copied";
      if (announcer) announcer.textContent = "Copied command to clipboard";
      clearTimeout(buttonEl._copyTimer);
      buttonEl._copyTimer = setTimeout(function () {
        buttonEl.classList.remove("is-copied");
        if (labelEl) labelEl.textContent = "Copy";
      }, 1800);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(showSuccess).catch(fallbackCopy);
    } else {
      fallbackCopy();
    }

    function fallbackCopy() {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy");
        showSuccess();
      } catch (_err) {
        /* ignore */
      }
      document.body.removeChild(area);
    }
  }

  function initInstallBar() {
    const tabs = Array.from(document.querySelectorAll(".install-tab"));
    const cmdText = document.getElementById("install-cmd-text");
    const noteText = document.getElementById("install-meta-note");
    const panel = document.getElementById("install-command-panel");
    const heroCopyBtn = document.getElementById("hero-copy-btn");
    const footerCopyBtn = document.getElementById("footer-copy-btn");

    function activateInstallTab(tab, focusTab) {
      tabs.forEach(function (t) {
        const active = t === tab;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-selected", active ? "true" : "false");
        t.tabIndex = active ? 0 : -1;
      });
      if (cmdText) cmdText.textContent = tab.getAttribute("data-install-cmd") || "";
      if (noteText) noteText.textContent = tab.getAttribute("data-install-note") || "";
      if (panel) panel.setAttribute("aria-labelledby", tab.id);
      if (focusTab) tab.focus();
    }

    tabs.forEach(function (tab, idx) {
      tab.tabIndex = idx === 0 ? 0 : -1;
      tab.addEventListener("click", function () {
        activateInstallTab(tab, false);
      });
      tab.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          const dir = e.key === "ArrowRight" ? 1 : -1;
          const next = tabs[(idx + dir + tabs.length) % tabs.length];
          activateInstallTab(next, true);
        }
      });
    });

    if (heroCopyBtn && cmdText) {
      heroCopyBtn.addEventListener("click", function () {
        copyToClipboard(cmdText.textContent || "", heroCopyBtn);
      });
    }

    if (footerCopyBtn) {
      footerCopyBtn.addEventListener("click", function () {
        const txt = footerCopyBtn.getAttribute("data-copy-text") || "";
        copyToClipboard(txt, footerCopyBtn);
      });
    }
  }

  /* ------------------------------------------------------------------------
     5. Hero Interactive Noah Code TUI Simulator
     ------------------------------------------------------------------------ */

  let activeScenario = "1";
  let currentDiffFile = "src/api/search.py";
  let isSplitDiff = true;

  function switchScenario(numStr, focusTab) {
    activeScenario = String(numStr);
    const tabs = Array.from(document.querySelectorAll(".scenario-tab"));
    tabs.forEach(function (tab) {
      const match = tab.getAttribute("data-scenario") === activeScenario;
      tab.classList.toggle("is-active", match);
      tab.setAttribute("aria-selected", match ? "true" : "false");
      tab.tabIndex = match ? 0 : -1;
      if (match && focusTab) tab.focus();
    });

    for (let i = 1; i <= 4; i++) {
      const panel = document.getElementById("tui-panel-" + i);
      if (panel) {
        const show = String(i) === activeScenario;
        panel.hidden = !show;
        panel.classList.toggle("is-active", show);
      }
    }

    if (activeScenario === "3") {
      const searchInput = document.getElementById("model-search-input");
      if (searchInput && focusTab === "input") {
        setTimeout(function () {
          searchInput.focus();
        }, 30);
      }
    }
  }

  function renderDiffPane() {
    const fileData = DIFF_FILES[currentDiffFile];
    const target = document.getElementById("diff-render-target");
    const fileHeader = document.getElementById("diff-active-filename");
    const hunkStatus = document.getElementById("diff-hunk-status");
    const stageLabel = document.getElementById("diff-stage-label");
    const revertLabel = document.getElementById("diff-revert-label");

    if (!fileData || !target) return;

    if (fileHeader) fileHeader.textContent = currentDiffFile;

    const stateLabel = fileData.reverted
      ? "Reverted (Press H to restore)"
      : fileData.staged
      ? "Staged (index)"
      : "Unstaged (worktree)";

    if (hunkStatus) {
      hunkStatus.textContent = fileData.hunkMeta + " · " + stateLabel;
    }
    if (stageLabel) {
      stageLabel.textContent = fileData.staged ? "Unstage Hunk" : "Stage Hunk";
    }
    if (revertLabel) {
      revertLabel.textContent = fileData.reverted ? "Undo Revert" : "Revert Hunk";
    }

    // Update sidebar scope tags (S vs U)
    document.querySelectorAll(".diff-file-item").forEach(function (btn) {
      const path = btn.getAttribute("data-diff-file");
      const selected = path === currentDiffFile;
      btn.classList.toggle("is-selected", selected);
      btn.setAttribute("aria-selected", selected ? "true" : "false");
      const tag = btn.querySelector(".diff-scope-tag");
      if (tag && path && DIFF_FILES[path]) {
        const st = DIFF_FILES[path].staged;
        tag.textContent = st ? "S" : "U";
        tag.classList.toggle("is-staged", st);
      }
    });

    if (fileData.reverted) {
      target.innerHTML =
        '<div class="diff-unified-row">' +
        '<span class="ln">44</span><span class="ln">44</span>' +
        '<span class="code">Hunk cleanly reverted in worktree · Checkpoint ◆ cp_94e1f0a intact (click H Undo Revert to restore).</span>' +
        "</div>";
      return;
    }

    if (isSplitDiff) {
      target.innerHTML = fileData.rows
        .map(function (r) {
          if (r.type === "ctx") {
            return (
              '<div class="diff-row diff-row-ctx">' +
              '<span class="ln">' + r.oldLn + "</span>" +
              '<span class="code">' + r.oldHtml + "</span>" +
              '<span class="ln">' + r.newLn + "</span>" +
              '<span class="code">' + r.newHtml + "</span>" +
              "</div>"
            );
          }
          if (r.type === "mod") {
            return (
              '<div class="diff-row diff-row-mod">' +
              '<span class="ln ln-del">' + r.oldLn + "</span>" +
              '<span class="code code-del">' + r.oldHtml + "</span>" +
              '<span class="ln ln-add">' + r.newLn + "</span>" +
              '<span class="code code-add">' + r.newHtml + "</span>" +
              "</div>"
            );
          }
          return (
            '<div class="diff-row diff-row-add-only">' +
            '<span class="ln ln-empty"></span>' +
            '<span class="code code-empty"></span>' +
            '<span class="ln ln-add">' + r.newLn + "</span>" +
            '<span class="code code-add">' + r.newHtml + "</span>" +
            "</div>"
          );
        })
        .join("");
    } else {
      const unifiedLines = [];
      fileData.rows.forEach(function (r) {
        if (r.type === "ctx") {
          unifiedLines.push(
            '<div class="diff-unified-row">' +
              '<span class="ln">' + r.oldLn + "</span>" +
              '<span class="ln">' + r.newLn + "</span>" +
              '<span class="code">' + r.oldHtml + "</span>" +
              "</div>"
          );
        } else if (r.type === "mod") {
          unifiedLines.push(
            '<div class="diff-unified-row">' +
              '<span class="ln ln-del">' + r.oldLn + "</span>" +
              '<span class="ln ln-del"></span>' +
              '<span class="code code-del">' + r.oldHtml + "</span>" +
              "</div>"
          );
          unifiedLines.push(
            '<div class="diff-unified-row">' +
              '<span class="ln ln-add"></span>' +
              '<span class="ln ln-add">' + r.newLn + "</span>" +
              '<span class="code code-add">' + r.newHtml + "</span>" +
              "</div>"
          );
        } else {
          unifiedLines.push(
            '<div class="diff-unified-row">' +
              '<span class="ln ln-add"></span>' +
              '<span class="ln ln-add">' + r.newLn + "</span>" +
              '<span class="code code-add">' + r.newHtml + "</span>" +
              "</div>"
          );
        }
      });
      target.innerHTML = unifiedLines.join("");
    }
  }

  function initCockpitSimulator() {
    const tabs = Array.from(document.querySelectorAll(".scenario-tab"));
    tabs.forEach(function (tab, idx) {
      tab.tabIndex = idx === 0 ? 0 : -1;
      tab.addEventListener("click", function () {
        switchScenario(tab.getAttribute("data-scenario") || "1", false);
      });
      tab.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          const dir = e.key === "ArrowRight" ? 1 : -1;
          const next = tabs[(idx + dir + tabs.length) % tabs.length];
          switchScenario(next.getAttribute("data-scenario") || "1", true);
        }
      });
    });

    // Global 1-4 shortcut keys when not typing in an input/textarea
    document.addEventListener("keydown", function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "textarea" || tag === "select" || (e.target && e.target.isContentEditable)) {
        return;
      }
      if (e.key === "1" || e.key === "2" || e.key === "3" || e.key === "4") {
        switchScenario(e.key, false);
      }
    });

    // Mode toggle (build <-> plan) in Noah header
    const modeBtn = document.getElementById("tui-mode-toggle");
    const modeLabel = document.getElementById("tui-active-mode");
    if (modeBtn && modeLabel) {
      modeBtn.addEventListener("click", function () {
        const isPlan = modeLabel.textContent === "build";
        modeLabel.textContent = isPlan ? "plan" : "build";
        modeBtn.classList.toggle("is-plan", isPlan);
      });
    }

    // Header model button jumps straight to Tab 3 (Model Picker)
    const headerModelBtn = document.getElementById("tui-header-model-btn");
    if (headerModelBtn) {
      headerModelBtn.addEventListener("click", function () {
        switchScenario("3", "input");
      });
    }

    // Panel 1: Subagent inspector pills
    const subChips = document.querySelectorAll("[data-agent-inspect]");
    const drawerTitle = document.getElementById("drawer-agent-title");
    const drawerOutput = document.getElementById("drawer-agent-output");
    subChips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        const key = chip.getAttribute("data-agent-inspect");
        if (!key || !SUBAGENT_TRACES[key]) return;
        subChips.forEach(function (c) {
          c.classList.toggle("is-selected", c === chip);
        });
        if (drawerTitle) drawerTitle.textContent = SUBAGENT_TRACES[key].title;
        if (drawerOutput) drawerOutput.textContent = SUBAGENT_TRACES[key].output;
      });
    });

    // Panel 1 Context Rail file click -> jump to Panel 2 Diff Review
    document.querySelectorAll("[data-jump-diff]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const file = btn.getAttribute("data-jump-diff");
        if (file && DIFF_FILES[file]) {
          currentDiffFile = file;
          renderDiffPane();
          switchScenario("2", false);
        }
      });
    });

    // Panel 2: Diff Screen interactions
    document.querySelectorAll(".diff-file-item").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const path = btn.getAttribute("data-diff-file");
        if (path && DIFF_FILES[path]) {
          currentDiffFile = path;
          renderDiffPane();
        }
      });
    });

    const splitBtn = document.getElementById("diff-split-btn");
    if (splitBtn) {
      splitBtn.addEventListener("click", function () {
        isSplitDiff = !isSplitDiff;
        splitBtn.classList.toggle("is-active", isSplitDiff);
        splitBtn.setAttribute("aria-pressed", isSplitDiff ? "true" : "false");
        splitBtn.innerHTML = isSplitDiff ? "<kbd>S</kbd> Split View" : "<kbd>S</kbd> Unified View";
        renderDiffPane();
      });
    }

    const stageBtn = document.getElementById("diff-stage-btn");
    if (stageBtn) {
      stageBtn.addEventListener("click", function () {
        const data = DIFF_FILES[currentDiffFile];
        if (!data) return;
        data.staged = !data.staged;
        renderDiffPane();
      });
    }

    const revertBtn = document.getElementById("diff-revert-btn");
    if (revertBtn) {
      revertBtn.addEventListener("click", function () {
        const data = DIFF_FILES[currentDiffFile];
        if (!data) return;
        data.reverted = !data.reverted;
        renderDiffPane();
      });
    }

    renderDiffPane();

    // Panel 4: /resume session selection & 1-click MCP toggles
    const resumeCards = document.querySelectorAll("[data-resume-id]");
    const resumeStatus = document.getElementById("resume-status-line");
    resumeCards.forEach(function (card) {
      card.addEventListener("click", function () {
        const id = card.getAttribute("data-resume-id");
        resumeCards.forEach(function (c) {
          const active = c === card;
          c.classList.toggle("is-selected", active);
          c.setAttribute("aria-selected", active ? "true" : "false");
        });
        if (id && RESUME_DETAILS[id] && resumeStatus) {
          resumeStatus.textContent = RESUME_DETAILS[id];
        }
      });
    });

    document.querySelectorAll("[data-mcp-toggle]").forEach(function (mcpBtn) {
      mcpBtn.addEventListener("click", function () {
        const connected = !mcpBtn.classList.contains("is-connected");
        mcpBtn.classList.toggle("is-connected", connected);
        mcpBtn.setAttribute("aria-pressed", connected ? "true" : "false");
        const pill = mcpBtn.querySelector(".mcp-state-pill");
        if (pill) {
          pill.textContent = connected ? "connected" : "1-click attach";
        }
      });
    });
  }

  /* ------------------------------------------------------------------------
     6. Panel 3: Zero-Typing Model Picker (Ctrl+L)
     ------------------------------------------------------------------------ */

  function initModelPicker() {
    const searchInput = document.getElementById("model-search-input");
    const listEl = document.getElementById("model-picker-list");
    const countEl = document.getElementById("model-match-count");
    const activeBadge = document.getElementById("picker-current-model");
    const headerModel = document.getElementById("tui-active-model");
    const filterPills = Array.from(document.querySelectorAll(".prov-filter-pill"));

    if (!listEl) return;

    let selectedModelId = "anthropic/claude-sonnet-5.5";
    let providerFilter = "all";
    let focusedIdx = 0;
    let currentFiltered = MODEL_CATALOG.slice();

    function escapeHtml(str) {
      return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function selectModel(modelObj) {
      selectedModelId = modelObj.id;
      if (activeBadge) activeBadge.textContent = modelObj.id;
      if (headerModel) headerModel.textContent = modelObj.shortName;
      renderModels();
    }

    function renderModels() {
      const q = searchInput ? searchInput.value.trim().toLowerCase() : "";
      currentFiltered = MODEL_CATALOG.filter(function (m) {
        const famOk = providerFilter === "all" || m.family === providerFilter;
        if (!famOk) return false;
        if (!q) return true;
        return (
          m.id.toLowerCase().indexOf(q) !== -1 ||
          m.desc.toLowerCase().indexOf(q) !== -1 ||
          m.ctx.toLowerCase().indexOf(q) !== -1
        );
      });

      if (focusedIdx >= currentFiltered.length) {
        focusedIdx = Math.max(0, currentFiltered.length - 1);
      }

      if (countEl) {
        countEl.textContent =
          currentFiltered.length + (currentFiltered.length === 1 ? " model" : " models");
      }

      if (currentFiltered.length === 0) {
        listEl.innerHTML =
          '<div class="picker-empty">No curated routes match — Noah also supports manual entry &amp; live OpenAI-compatible `/v1/models` discovery.</div>';
        return;
      }

      listEl.innerHTML = currentFiltered
        .map(function (m, idx) {
          const isSel = m.id === selectedModelId;
          const isFoc = idx === focusedIdx;
          const classes =
            "model-option-btn" +
            (isSel ? " is-selected" : "") +
            (isFoc ? " is-focused" : "");
          return (
            '<button type="button" class="' +
            classes +
            '" role="option" aria-selected="' +
            (isSel ? "true" : "false") +
            '" data-model-index="' +
            idx +
            '">' +
            '<div class="model-opt-top">' +
            '<span class="model-opt-id">' +
            (isSel ? "✓ " : "") +
            escapeHtml(m.id) +
            "</span>" +
            '<span class="model-opt-ctx">' +
            escapeHtml(m.ctx) +
            "</span>" +
            "</div>" +
            '<div class="model-opt-desc">' +
            escapeHtml(m.desc) +
            "</div>" +
            "</button>"
          );
        })
        .join("");

      listEl.querySelectorAll(".model-option-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          const idx = Number(btn.getAttribute("data-model-index"));
          if (!Number.isNaN(idx) && currentFiltered[idx]) {
            focusedIdx = idx;
            selectModel(currentFiltered[idx]);
          }
        });
      });
    }

    if (searchInput) {
      searchInput.addEventListener("input", function () {
        focusedIdx = 0;
        renderModels();
      });

      searchInput.addEventListener("keydown", function (e) {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          if (currentFiltered.length > 0) {
            focusedIdx = (focusedIdx + 1) % currentFiltered.length;
            renderModels();
          }
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          if (currentFiltered.length > 0) {
            focusedIdx = (focusedIdx - 1 + currentFiltered.length) % currentFiltered.length;
            renderModels();
          }
        } else if (e.key === "Enter") {
          e.preventDefault();
          if (currentFiltered[focusedIdx]) {
            selectModel(currentFiltered[focusedIdx]);
          }
        }
      });
    }

    filterPills.forEach(function (pill) {
      pill.addEventListener("click", function () {
        providerFilter = pill.getAttribute("data-prov-filter") || "all";
        filterPills.forEach(function (p) {
          p.classList.toggle("is-active", p === pill);
        });
        focusedIdx = 0;
        renderModels();
      });
    });

    renderModels();
  }

  /* ------------------------------------------------------------------------
     7. Interactive Keyboard & Slash Command Explorer
     ------------------------------------------------------------------------ */

  function initShortcutExplorer() {
    const grid = document.getElementById("shortcut-grid");
    const searchInput = document.getElementById("shortcut-search-input");
    const catTabs = Array.from(document.querySelectorAll(".explorer-tab"));
    if (!grid) return;

    let activeCat = "all";

    function escapeHtml(str) {
      return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function renderShortcuts() {
      const q = searchInput ? searchInput.value.trim().toLowerCase() : "";
      const matches = SHORTCUTS_AND_COMMANDS.filter(function (item) {
        const catOk = activeCat === "all" || item.cat === activeCat;
        if (!catOk) return false;
        if (!q) return true;
        return (
          item.key.toLowerCase().indexOf(q) !== -1 ||
          item.scope.toLowerCase().indexOf(q) !== -1 ||
          item.desc.toLowerCase().indexOf(q) !== -1
        );
      });

      if (matches.length === 0) {
        grid.innerHTML =
          '<div class="picker-empty">No matching controls — inside Noah Code, press `?` or `Ctrl+P` to search every setting and command.</div>';
        return;
      }

      grid.innerHTML = matches
        .map(function (item) {
          return (
            '<div class="shortcut-card">' +
            '<div class="shortcut-card-top">' +
            '<kbd class="shortcut-key">' +
            escapeHtml(item.key) +
            "</kbd>" +
            '<span class="shortcut-scope">' +
            escapeHtml(item.scope) +
            "</span>" +
            "</div>" +
            '<p class="shortcut-desc">' +
            escapeHtml(item.desc) +
            "</p>" +
            "</div>"
          );
        })
        .join("");
    }

    catTabs.forEach(function (tab, idx) {
      tab.tabIndex = idx === 0 ? 0 : -1;
      tab.addEventListener("click", function () {
        activeCat = tab.getAttribute("data-cmd-cat") || "all";
        catTabs.forEach(function (t) {
          const active = t === tab;
          t.classList.toggle("is-active", active);
          t.setAttribute("aria-selected", active ? "true" : "false");
          t.tabIndex = active ? 0 : -1;
        });
        renderShortcuts();
      });
      tab.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          const dir = e.key === "ArrowRight" ? 1 : -1;
          const next = catTabs[(idx + dir + catTabs.length) % catTabs.length];
          next.click();
          next.focus();
        }
      });
    });

    if (searchInput) {
      searchInput.addEventListener("input", renderShortcuts);
    }

    renderShortcuts();
  }

  /* ------------------------------------------------------------------------
     Boot
     ------------------------------------------------------------------------ */
  document.addEventListener("DOMContentLoaded", function () {
    initTheme();
    initMobileNav();
    initInstallBar();
    initCockpitSimulator();
    initModelPicker();
    initShortcutExplorer();
  });
})();
