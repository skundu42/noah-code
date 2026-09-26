"""A keyboard-first launcher for the built-in team workflows."""

from __future__ import annotations

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, OptionList, Static
from textual.widgets.option_list import Option

from noah_code.teams import get_team_workflow, team_workflows
from noah_code.themes import THEMES


class TeamLauncherScreen(ModalScreen[str | None]):
    """Choose a workflow and return it to the editable prompt composer."""

    BINDINGS = [Binding("escape,f9", "close", "Close", show=True)]

    def __init__(self, *, mode: str = "build") -> None:
        super().__init__()
        self.mode = mode
        self._workflow: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="team-dialog"):
            yield Label("Work with a team", id="team-title")
            yield Static(
                "Choose a workflow, then describe the outcome in your prompt.",
                id="team-intro",
            )
            yield OptionList(id="team-workflows", compact=True)
            yield Static("", id="team-preview")
            with Horizontal(id="team-buttons"):
                yield Button("Use workflow", id="team-use", variant="primary")
                yield Button("Cancel", id="team-cancel")
            yield Static("↑/↓ choose · Enter use · Esc close", id="team-hint")

    def on_mount(self) -> None:
        self.set_class(self.size.width < 100, "narrow")
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        choices = self.query_one("#team-workflows", OptionList)
        for workflow in team_workflows():
            disabled = self.mode == "plan" and not workflow.readonly
            access = "read-only" if workflow.readonly else "can edit files"
            if disabled:
                access = "switch to Build mode to use"
            label = Text(workflow.title, style=f"bold {palette.accent}")
            label.append(f"  {access}\n", style=palette.muted)
            label.append(workflow.description, style=palette.text)
            choices.add_option(Option(label, id=workflow.name, disabled=disabled))
        choices.action_first()
        choices.focus()

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 100, "narrow")

    @on(OptionList.OptionHighlighted, "#team-workflows")
    def _highlight(self, event: OptionList.OptionHighlighted) -> None:
        if not event.option.id:
            return
        self._workflow = event.option.id
        workflow = get_team_workflow(self._workflow)
        palette = getattr(self.app, "theme_palette", THEMES["atom-one-dark"])
        text = Text()
        for index, phase in enumerate(workflow.phases):
            if index:
                text.append("\n")
            text.append(f"{index + 1}. {phase.title}", style=f"bold {palette.text}")
            text.append(
                "  " + " + ".join(role.agent for role in phase.roles), style=palette.muted
            )
        text.append(
            "\n\nRead-only teammates can run together. Each phase hands its findings "
            "to the next. A failure or request for input stops the workflow.",
            style=palette.muted,
        )
        self.query_one("#team-preview", Static).update(text)

    @on(OptionList.OptionSelected, "#team-workflows")
    def _select(self, event: OptionList.OptionSelected) -> None:
        if event.option.id and not event.option.disabled:
            self.dismiss(event.option.id)

    @on(Button.Pressed, "#team-use")
    def _use(self) -> None:
        if self._workflow:
            workflow = get_team_workflow(self._workflow)
            if self.mode != "plan" or workflow.readonly:
                self.dismiss(self._workflow)

    @on(Button.Pressed, "#team-cancel")
    def action_close(self) -> None:
        self.dismiss(None)
