"""Scope rules shared by every store and the runtime. One place, so search,
decision checks and the handoff filter cannot disagree about who sees what.

Scopes widen in order personal < team < org. v0.1 has one org.
"""

from .contract import Caller, Event, Knowledge, Scope

ORG_OWNER = "*"


def event_visible(event: Event, caller: Caller, scope: Scope) -> bool:
    """personal: the caller's own events.
    team: plus the team-scoped events of the caller's team.
    org: plus every org-scoped event."""
    if event.user == caller.user:
        return True
    if scope in ("team", "org") and event.scope == "team" and event.team == caller.team:
        return True
    return scope == "org" and event.scope == "org"


def knowledge_visible(item: Knowledge, caller: Caller, scope: Scope) -> bool:
    if item.scope == "personal":
        return item.owner == caller.user
    if item.scope == "team":
        return scope in ("team", "org") and item.owner == caller.team
    return scope == "org"


def owner_for(event: Event) -> str:
    """The owner a knowledge item gets when it is shared exactly as widely as event."""
    if event.scope == "personal":
        return event.user
    if event.scope == "team":
        return event.team
    return ORG_OWNER


def audience_sees(event: Event, scope: Scope, owner: str) -> bool:
    """Whether everyone in the audience of an item (scope, owner) may see event.
    A promoted item is shown to its whole audience, so every source must pass.

    Conservative on purpose: a personal item may cite its owner's events and
    org events, but not team events, because an item does not record its
    owner's team."""
    if event.scope == "org":
        return True
    if scope == "personal":
        return event.user == owner
    if scope == "team":
        return event.scope == "team" and event.team == owner
    return False
