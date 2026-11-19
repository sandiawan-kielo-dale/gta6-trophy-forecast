"""GTA VI week-one trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Python 3.13, standard library only. Run: python3 -P -s main.py
Reads every *.in in $MAKEFUN_INPUT_DIR (default /input), writes <id>.out to $MAKEFUN_OUTPUT_DIR (default /output).

Model, in one paragraph: a trophy's earn rate on Exophase is earners / players tracked. Players join the tracked
population over time (a player is tracked once they hold the game's first trophy), and each player then progresses
through the game at their own pace. So the number of earners of trophy k at time t is

    E_k(t) = A_k + h * (N(t) - N_0) + c_k * sum_j n_j * Phi((ln(t - s_j) - mu_k) / sigma)

where n_j players joined at time s_j (read off the history's players_tracked growth), A_k is the launch-night
population that already held the trophy (review copies and the like), h the share of new players who unlock
everything at once (cheaters, measured from the platinum, which nobody can earn legitimately in week one), c_k the
share of players who ever go for the trophy and mu_k the log of the wall-clock hours a typical player needs to reach
it. mu_k is fitted per trophy to the case's history, with a prior built from where the trophy sits in the game
(story missions in chapter order, side content by effort) shifted by a pace learnt from the well-measured trophies.
Future joins extrapolate the recent join rate with a decay. The forecast is a mixture over scenarios (sigma, join
decay, how stale the answering page is), each a logit-normal around the model's rate, and the quantiles are read off
that mixture. Earners never decrease, which floors every scenario.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZQ = [-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536]

# Share of wall-clock time a typical tracked player spends in the game during launch week (prior; refined by data).
PLAY_SHARE = 0.2
# Hours over which the first read's players are assumed to have joined before it.
FIRST_COHORT_SPAN = 6.0
# Widest cohort bin, in hours.
BIN_HOURS = 3.0

# Priors per trophy, keyed by the normalised description: (hours of play a typical player needs, share who ever
# get it, prior sd of mu in ln units). Story positions come from the published mission order (93 missions:
# chapter 1 ends at 22, Breakfast Run 30, chapter 2 ends 35, The Cartel Heist ends chapter 3 at 61, Child's Play 78,
# Headliners 79, VCIA Takedown 84, chapter 4 ends 86, Operation Savage Glory 91, the finale 93) over a ~62 hour story.
PRIORS = {
    "completegonefishing": (1.0, 1.0, 0.5),
    # Ever-reach shares follow Red Dead Redemption 2's lifetime story curve among trophy-site players
    # (chapter 1 ~84%, chapter 2 ~63%, chapter 3 ~55%, chapter 4 ~50%, the end ~46%), a little higher for GTA.
    "completechapter1": (10.0, 0.86, 0.6),
    "completebreakfastrun": (20.0, 0.78, 0.6),
    "completechapter2": (23.0, 0.75, 0.6),
    "completechapter3": (40.0, 0.66, 0.6),
    "completethecartelheist": (40.0, 0.66, 0.6),
    "completechildsplay": (52.0, 0.61, 0.6),
    "completeheadliners": (52.5, 0.61, 0.6),
    "completevciatakedown": (56.0, 0.59, 0.6),
    "completechapter4": (57.0, 0.58, 0.6),
    "completeoperationsavageglory": (60.5, 0.56, 0.6),
    "completethefinalmission": (62.0, 0.55, 0.6),
    "attain100completion": (180.0, 0.1, 1.0),
    "complete250stunts": (90.0, 0.15, 1.2),
    "tryeachtypeofactivityonce": (40.0, 0.4, 1.2),
    "earnagoldmedalinallshootingrangechallenges": (60.0, 0.2, 1.2),
    "winatopfundroneraceateverylocation": (70.0, 0.2, 1.2),
    "win20differentraces": (50.0, 0.3, 1.2),
    "beatcalinafishingchallenge": (4.0, 0.8, 1.0),
    "completeanarcadegame": (12.0, 0.5, 1.2),
    "completethefinalmissionforandy": (35.0, 0.5, 1.2),
    "completethefinalmissionforkenton": (35.0, 0.5, 1.2),
    "completethefinalmissionforbilly": (35.0, 0.5, 1.2),
    "complete8missionstrandsforstrangersacrossthestateofleonida": (55.0, 0.35, 1.2),
    "completethefinalmissionforrandy": (35.0, 0.5, 1.2),
    "completethefinalmissionfortyrone": (35.0, 0.5, 1.2),
    "meditateateveryspotacrossleonida": (70.0, 0.2, 1.2),
    "huntauniqueanimalforbret": (30.0, 0.5, 1.2),
    "complete3collectiblecardsets": (60.0, 0.2, 1.2),
    "wainkitat100locations": (80.0, 0.15, 1.2),
    "complete50challenges": (70.0, 0.2, 1.2),
    "losea4starwantedlevelwithyourpartnerandescapetheheat": (15.0, 0.7, 1.2),
    "rob3shopsbacktobackwithyourpartnerwithoutlosingyourwantedlevel": (15.0, 0.6, 1.2),
    "discover100vehicles": (12.0, 0.8, 1.2),
    "takedown5enemiesinasingleuseofcombatfocus": (12.0, 0.7, 1.2),
    "makeanunscheduleddepartureatvcia": (56.0, 0.8, 1.0),
    "kindahotintheserhinos": (30.0, 0.6, 1.2),
    "robajewelrystorewithoutbeingdetected": (30.0, 0.5, 1.2),
    "rob5bankvaultsandescape": (45.0, 0.4, 1.2),
    "robanarmoredtruckandescape": (20.0, 0.6, 1.2),
    "exploreallofleonida": (50.0, 0.4, 1.2),
    "robanillegaloperationbelongingtoeachgang": (45.0, 0.4, 1.2),
    "enjoyacocktailwithaviewonoceanbeach": (42.0, 0.8, 1.0),
    "smashyourwayoutofavehicleshowroominclassicstyle": (40.0, 0.4, 1.2),
    "followthroughatthekoltassociatesbowlingalley": (40.0, 0.5, 1.2),
    "apartnershipwithmassappeal": (45.0, 0.6, 1.2),
    "choicesmadeforbetterorworse": (60.0, 0.8, 1.0),
    "secondtimesthecharm": (45.0, 0.6, 1.2),
    "whatthehelljusthappened": (50.0, 0.3, 1.2),
    "bowlemoverwithameleedriveby": (45.0, 0.3, 1.2),
    "earn250000launderingmarkedbillsthroughyourcontacts": (35.0, 0.6, 1.2),
    "spend100000acrossallshops": (30.0, 0.6, 1.2),
    "purchaseallgarageproperties": (60.0, 0.3, 1.2),
    "register10vehiclestothescooterbrosapp": (30.0, 0.5, 1.2),
}
TIER_PRIORS = {
    "platinum": (200.0, 0.1, 1.0),
    "gold": (100.0, 0.2, 1.2),
    "silver": (50.0, 0.35, 1.2),
    "bronze": (30.0, 0.5, 1.5),
}

# Scenario grids: (value, prior weight). Sigma weights are then updated by the fit (SIGMA_TEMPER tempers it).
SIGMAS = [(0.55, 0.2), (0.75, 0.3), (0.95, 0.3), (1.2, 0.2)]
SIGMA_TEMPER = 3.0
# Overall scale on each scenario's logit spread (TROPHY_FC_SPREAD overrides it for local calibration runs only).
SPREAD = float(os.environ.get("TROPHY_FC_SPREAD", "0.85"))
BETAS = [(0.5, 0.3), (1.0, 0.4), (1.6, 0.3)]
LAG_FRACS = [(0.15, 0.34), (0.5, 0.33), (0.85, 0.33)]
STORY_KEYS = {
    "completechapter1", "completebreakfastrun", "completechapter2", "completechapter3", "completethecartelheist",
    "completechildsplay", "completeheadliners", "completevciatakedown", "completechapter4",
    "completeoperationsavageglory", "completethefinalmission",
}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_key(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() / 3600.0


def prior_for(trophy):
    key = norm_key(trophy.get("description"))
    if key in PRIORS:
        return PRIORS[key]
    return TIER_PRIORS.get(trophy.get("tier"), TIER_PRIORS["bronze"])


def snapshots_of(history, k_count):
    """Collapse the reads into distinct page states: (time first seen, players, rates)."""
    snaps = []
    for read in history:
        try:
            t = parse_time(read["at"])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        n = read.get("players_tracked")
        rates = list(read.get("rates") or [])
        rates = (rates + [None] * k_count)[:k_count]
        if not isinstance(n, (int, float)) or n <= 0:
            n = snaps[-1][1] if snaps else None
            if n is None:
                continue
        state = (float(n), tuple(rates))
        if snaps and (snaps[-1][1], snaps[-1][2]) == state:
            continue
        snaps.append((t, float(n), tuple(rates)))
    return snaps


def build_cohorts(snaps):
    """Join cohorts (time, count) implied by the players-tracked path."""
    t0, n0 = snaps[0][0], snaps[0][1]
    cohorts = []
    parts = max(1, int(math.ceil(FIRST_COHORT_SPAN / BIN_HOURS)))
    for i in range(parts):
        cohorts.append((t0 - FIRST_COHORT_SPAN * (i + 0.5) / parts, n0 / parts))
    for (ta, na, _), (tb, nb, _) in zip(snaps, snaps[1:]):
        dn = nb - na
        if dn <= 0 or tb <= ta:
            continue
        parts = max(1, int(math.ceil((tb - ta) / BIN_HOURS)))
        for i in range(parts):
            cohorts.append((ta + (tb - ta) * (i + 0.5) / parts, dn / parts))
    # Merge into BIN_HOURS-wide buckets (count-weighted mean join time), so frequent refreshes stay cheap.
    buckets = {}
    for s, n in cohorts:
        b = math.floor((s - t0) / BIN_HOURS)
        acc = buckets.setdefault(b, [0.0, 0.0])
        acc[0] += s * n
        acc[1] += n
    return [(v[0] / v[1], v[1]) for _, v in sorted(buckets.items()) if v[1] > 0]



class CaseModel:
    def __init__(self, case):
        self.case = case
        self.trophies = case["trophies"]
        self.k = len(self.trophies)
        self.snaps = snapshots_of(case.get("history") or [], self.k)
        self.t_at = parse_time(case["at"])
        self.t_target = parse_time(case["target"]) if case.get("target") else self.t_at + case["horizon_minutes"] / 60.0

    def last_rate(self, i):
        for _, _, rates in reversed(self.snaps):
            if rates[i] is not None:
                return float(rates[i])
        return None


def mixture_quantiles(components):
    """components: list of (weight, mean, sd) on the logit scale. Returns the five quantiles (logit)."""
    tot = sum(w for w, _, _ in components)
    comps = [(w / tot, m, max(sd, 1e-3)) for w, m, sd in components if w > 0]
    lo = min(m - 6 * sd for _, m, sd in comps)
    hi = max(m + 6 * sd for _, m, sd in comps)
    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            cdf = sum(w * phi((mid - m) / sd) for w, m, sd in comps)
            if cdf < tau:
                a = mid
            else:
                b = mid
        out.append(0.5 * (a + b))
    return out


def round4(x):
    v = math.floor(x * 10000 + 0.5) / 10000
    return int(v) if v.is_integer() else v


def finish_row(tid, qs_pct):
    qs = [min(100.0, max(0.0, q)) for q in qs_pct]
    rounded = []
    for q in qs:
        r = round4(q)
        if rounded and r < rounded[-1]:
            r = rounded[-1]
        rounded.append(r)
    return {"id": tid, "q": rounded}


def baseline_row(tid, rate, days):
    lg = logit(rate if rate is not None else 0.5)
    sp = 0.5 * math.sqrt(max(days, 1e-6)) + 0.3
    return finish_row(tid, [inv_logit(lg + z * sp) for z in ZQ])


def forecast_case(case):
    model = CaseModel(case)
    snaps = model.snaps
    rows = []
    horizon_h = max(0.0, model.t_target - model.t_at)
    if not snaps:
        for tr in model.trophies:
            rows.append(finish_row(tr["id"], [0.01, 0.1, 0.5, 2.0, 10.0]))
        return {"trophies": rows}

    t0, n0 = snaps[0][0], snaps[0][1]
    t_last, n_last = snaps[-1][0], snaps[-1][1]
    cohorts = build_cohorts(snaps)

    # Refresh cadence: mean gap between page changes, and how long the page has sat unchanged.
    gaps = [b[0] - a[0] for a, b in zip(snaps, snaps[1:])]
    since_change = max(0.0, model.t_at - t_last)
    gap = max(sum(gaps) / len(gaps) if gaps else 3.0, since_change, 0.5)
    p_stale = min(0.5, math.exp(-max(horizon_h, 0.0) / gap)) if horizon_h > 0 else 1.0

    # Recent join rate per hour, and the game's age at the middle of the window it was measured over.
    t_launch = t0 - FIRST_COHORT_SPAN
    win_start = max(t0, t_last - 24.0)
    base = snaps[0]
    for s in snaps:
        if s[0] <= win_start:
            base = s
    if t_last - base[0] > 0.5 and n_last > base[1]:
        lam = (n_last - base[1]) / (t_last - base[0])
        age_ref = 0.5 * (base[0] + t_last) - t_launch
    else:
        lam = 0.25 * n_last
        age_ref = t_last - t_launch

    # Cheater share: platinum earners gained per new player.
    hack = 0.0
    for i, tr in enumerate(model.trophies):
        if tr.get("tier") == "platinum":
            first = next((s for s in snaps if s[2][i] is not None), None)
            if first is not None and n_last > first[1]:
                e_first = first[2][i] * first[1] / 100.0
                e_last = (model.last_rate(i) or 0.0) * n_last / 100.0
                hack = max(0.0, min(0.003, (e_last - e_first) / (n_last - first[1])))

    # Fitting window: snapshots of the last 48 hours, at most 24 of them.
    fit_snaps = [s for s in snaps if s[0] >= t_last - 48.0]
    if len(fit_snaps) > 24:
        step = len(fit_snaps) / 24.0
        fit_snaps = [fit_snaps[int(i * step)] for i in range(24)] + [fit_snaps[-1]]
        dedup = []
        for s in fit_snaps:
            if not dedup or dedup[-1] is not s:
                dedup.append(s)
        fit_snaps = dedup

    info = []
    for i, tr in enumerate(model.trophies):
        h_need, c_k, tau = prior_for(tr)
        first = next((s for s in snaps if s[2][i] is not None), None)
        a_k = first[2][i] * first[1] / 100.0 if first is not None else 0.0
        n_first = first[1] if first is not None else n0
        obs = [(s[0], s[1], s[2][i]) for s in fit_snaps if s[2][i] is not None]
        info.append({"h": h_need, "c": c_k, "tau": tau, "a": a_k, "nf": n_first, "obs": obs,
                     "emin": max(0.05, 0.4 * h_need / 0.75), "story": norm_key(tr.get("description")) in STORY_KEYS})

    def fit_mu(inf, sigma, mu_prior, tau):
        obs = inf["obs"]
        if not obs:
            return mu_prior, tau, 0.0
        c_k, a_k, nf, emin = inf["c"], inf["a"], inf["nf"], inf["emin"]
        # Pre-computed log exposures of every cohort at every observed snapshot, and each term's fixed parts.
        terms = []
        for t, n, r in obs:
            les = [(math.log(t - s), cn) for s, cn in cohorts if t - s > emin]
            e_obs = r * n / 100.0
            # count noise, the page's 2-decimal rounding, and model misfit
            sd2 = 1.0 / max(e_obs, 1.0) + (0.0029 / max(r, 0.005)) ** 2 + 0.12 ** 2
            w = math.exp(-(t_last - t) / 24.0) / sd2
            terms.append((les, a_k + hack * max(0.0, n - nf), n, logit(r), w))
        inv_s = 1.0 / sigma / math.sqrt(2.0)

        def loss(mu):
            total = ((mu - mu_prior) / tau) ** 2
            for les, fixed, n, lr, w in terms:
                reg = 0.0
                for le, cn in les:
                    reg += cn * math.erfc((mu - le) * inv_s)
                pred = 100.0 * (fixed + 0.5 * c_k * reg) / n
                total += w * (logit(pred) - lr) ** 2
            return total

        grid = [mu_prior + d * 1.0 for d in range(-6, 7)]
        vals = [loss(m) for m in grid]
        j = min(range(len(grid)), key=lambda x: vals[x])
        a, b = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
        g = (math.sqrt(5) - 1) / 2
        x1, x2 = b - g * (b - a), a + g * (b - a)
        f1, f2 = loss(x1), loss(x2)
        for _ in range(12):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - g * (b - a)
                f1 = loss(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + g * (b - a)
                f2 = loss(x2)
        mu_hat = 0.5 * (a + b)
        hstep = 0.15
        f0 = loss(mu_hat)
        curv = (loss(mu_hat + hstep) - 2 * f0 + loss(mu_hat - hstep)) / (hstep * hstep)
        sd = math.sqrt(2.0 / curv) if curv > 1e-9 else tau
        return mu_hat, min(sd, tau), f0

    results = []
    for sigma, w_sigma in SIGMAS:
        # Pass 1: the pace shift from trophies with plenty of regular earners, story ones first (their
        # positions in the game are known; side-content effort is a guess).
        shifts, story_shifts = [], []
        for inf in info:
            if not inf["obs"]:
                continue
            r_last = inf["obs"][-1][2]
            e_last = r_last * inf["obs"][-1][1] / 100.0
            if r_last >= 99.99 or e_last - inf["a"] < 20:
                continue
            mu0 = math.log(inf["h"] / PLAY_SHARE)
            mu_hat, _, _ = fit_mu(inf, sigma, mu0, 3.0)
            (story_shifts if inf["story"] else shifts).append(mu_hat - mu0)
        pool = story_shifts if story_shifts else shifts
        shift = sorted(pool)[len(pool) // 2] if pool else 0.0
        shift = max(-2.0, min(2.0, shift))
        fits = []
        total_loss = 0.0
        for inf in info:
            mu0 = math.log(inf["h"] / PLAY_SHARE) + shift
            mu_hat, mu_sd, f = fit_mu(inf, sigma, mu0, inf["tau"])
            fits.append((mu_hat, mu_sd))
            total_loss += f
        results.append([sigma, w_sigma, fits, total_loss])

    # Weight the sigma scenarios by how well each fits the history (tempered, so no single one takes over).
    best = max(1e-9, min(r[3] for r in results))
    for r in results:
        r[1] = r[1] * math.exp(-SIGMA_TEMPER * (r[3] / best - 1.0))
    wsum = sum(r[1] for r in results)
    for r in results:
        r[1] /= wsum
    if os.environ.get("TROPHY_FC_DEBUG"):
        print(f"lam={lam:.1f}/h age_ref={age_ref:.1f}h hack={hack:.6f} gap={gap:.2f} p_stale={p_stale:.3f} "
              f"sigma_w={[(r[0], round(r[1], 3), round(r[3], 1)) for r in results]}", file=sys.stderr)

    # Each (join decay, page staleness) scenario's population at its effective time: log exposures and size.
    pop_cache = {}

    def population(t_eff, beta):
        key = (round(t_eff, 6), beta)
        if key not in pop_cache:
            allc = list(cohorts)
            n_future = 0.0
            if t_eff > t_last:
                parts = max(1, int(math.ceil((t_eff - t_last) / BIN_HOURS)))
                width = (t_eff - t_last) / parts
                for p in range(parts):
                    s = t_last + width * (p + 0.5)
                    rate = lam * (age_ref / max(s - t_launch, age_ref * 0.5)) ** beta
                    allc.append((s, rate * width))
                    n_future += rate * width
            les = [(math.log(t_eff - s), cn) for s, cn in allc if t_eff - s > 0.05]
            pop_cache[key] = (les, n_last + n_future)
        return pop_cache[key]

    def predict_rate(inf, sigma, mu, t_eff, beta):
        les, n_t = population(t_eff, beta)
        lemin = math.log(inf["emin"])
        inv_s = 1.0 / sigma / math.sqrt(2.0)
        reg = 0.0
        for le, cn in les:
            if le > lemin:
                reg += cn * math.erfc((mu - le) * inv_s)
        e = inf["a"] + hack * max(0.0, n_t - inf["nf"]) + inf["c"] * 0.5 * reg
        return 100.0 * e / n_t, n_t

    for i, tr in enumerate(model.trophies):
        tid = tr["id"]
        try:
            r_last = model.last_rate(i)
            if r_last is None:
                rows.append(finish_row(tid, [0.01, 0.1, 0.5, 2.0, 10.0]))
                continue
            if r_last >= 100.0:
                rows.append(finish_row(tid, [99.97, 99.99, 100.0, 100.0, 100.0]))
                continue
            e_floor = r_last * n_last / 100.0
            inf = info[i]
            comps = []
            if p_stale > 0:
                comps.append((p_stale, logit(r_last), 0.03))
            for sigma, w_sigma, fits, _ in results:
                mu, mu_sd = fits[i]
                for beta, w_beta in BETAS:
                    for lag_frac, w_lag in LAG_FRACS:
                        t_eff = max(t_last, model.t_target - lag_frac * min(gap, 36.0))
                        pr, n_t = predict_rate(inf, sigma, mu, t_eff, beta)
                        pr = max(pr, 100.0 * e_floor / n_t)
                        up, _ = predict_rate(inf, sigma, mu - mu_sd, t_eff, beta)
                        dn, _ = predict_rate(inf, sigma, mu + mu_sd, t_eff, beta)
                        dn = max(dn, 100.0 * e_floor / n_t)
                        sd_par = 0.5 * abs(logit(up) - logit(dn))
                        move = abs(logit(pr) - logit(r_last))
                        sd_mod = 0.12 + 0.08 * math.sqrt(horizon_h / 24.0) + 0.15 * move
                        sd = SPREAD * math.sqrt(sd_par ** 2 + sd_mod ** 2)
                        comps.append(((1 - p_stale) * w_sigma * w_beta * w_lag, logit(pr), sd))
            qs = mixture_quantiles(comps)
            rows.append(finish_row(tid, [inv_logit(v) for v in qs]))
        except Exception:
            rows.append(baseline_row(tid, model.last_rate(i), horizon_h / 24.0))
    # Story trophies are nested (whoever finished a later mission finished every earlier one), so no quantile of a
    # later story trophy may exceed the same quantile of an earlier one.
    story = sorted((prior_for(tr)[0], i) for i, tr in enumerate(model.trophies)
                   if norm_key(tr.get("description")) in STORY_KEYS and i < len(rows))
    cap = None
    for _, i in story:
        q = rows[i]["q"]
        if cap is not None:
            q = [min(a, b) for a, b in zip(q, cap)]
            rows[i] = {"id": rows[i]["id"], "q": q}
        cap = q
    return {"trophies": rows}


def safe_forecast(case):
    try:
        return forecast_case(case)
    except Exception:
        rows = []
        hist = case.get("history") or []
        last = hist[-1].get("rates") if hist else None
        days = (case.get("horizon_minutes") or 1440) / 1440.0
        for i, tr in enumerate(case.get("trophies") or []):
            r = last[i] if last and i < len(last) else None
            rows.append(baseline_row(tr["id"], r, days))
        return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = safe_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
