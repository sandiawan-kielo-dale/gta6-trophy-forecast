# GTA VI week-one trophy rarity forecast

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sandiawan-kielo-dale/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)

[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sandiawan-kielo-dale/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

An entry for make.fun's *Forecast how many GTA VI players earn each trophy in week one*: for every PS5 trophy of
Grand Theft Auto VI, five quantiles of its Exophase earn rate on 2026-11-26 00:00 UTC, scored by pinball loss on the
logit scale.

## Method

A rate on Exophase is *earners / players tracked*, and both halves move in launch week: casual players keep joining
(so early trophies' rates fall) while everyone already playing works through the story (so later trophies rise from
almost nothing). `main.py` models the two halves separately instead of extrapolating each rate on its own.

1. **Page states.** Exophase refreshes a game's counts on its own schedule, so consecutive identical reads are merged
   into one page state, stamped with the time it was first seen. The mean gap between refreshes is measured too.
2. **Join cohorts.** The players-tracked path gives how many players joined in each 3-hour bucket. Future joins
   extrapolate the last 24 hours' join rate with a power-law decay in the game's age (three decay scenarios).
3. **Progress.** A player who joined at time *s* holds trophy *k* at time *t* with probability
   `c_k * Phi((ln(t - s) - mu_k) / sigma)`: `mu_k` is the log of the wall-clock hours a typical player needs, `c_k`
   the share who ever go for it, `sigma` the spread of pace across players. Nobody can reach a trophy faster than an
   18-hour-a-day speedrun allows.
4. **Floors.** The launch-night population (review copies: four platinums before release) keeps what it had, and a
   cheater share measured from the platinum's growth (it can't be earned legitimately in week one) unlocks
   everything. Earners never decrease, so the last observed earner count divided by the forecast population floors
   every forecast.
5. **Priors.** Each trophy's prior `mu_k` comes from where it sits in the game: story trophies by the published mission
   order (93 missions; chapter 1 ends at 22, Breakfast Run 30, chapter 2 at 35, The Cartel Heist closes chapter 3 at
   61, Child's Play 78, Headliners 79, VCIA Takedown 84, chapter 4 at 86, Operation Savage Glory 91, the finale 93,
   over a ~60-80 hour story), side content by its stated effort (platinum ~200 h). `c_k` for the story follows Red
   Dead Redemption 2's lifetime chapter curve. A common pace shift is learnt from the well-measured story trophies,
   then every `mu_k` is fitted to its own history (recent page states weighted most, noise from counts and the page's
   2-decimal rounding), shrunk toward its prior.
6. **Forecast.** A mixture over `sigma` (weighted by how well each fits the history), join decay and how stale the
   answering page may be, each component a logit-normal whose spread combines the fit's uncertainty and a model
   error growing with the horizon and the size of the move. Story trophies are nested, so a later mission's quantiles
   never exceed an earlier one's. A trophy at 100% (the first one, which defines being tracked) stays there.

Synthetic launch weeks (simulated players with heterogeneous, decaying play time, regional join waves, cheaters,
review copies and a randomly refreshing page) were used to test and calibrate it.

## Run

Python 3.13, standard library only:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py

It reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR`
(default `/output`). It runs in well under a second per case and is deterministic.

## License

MIT, see [LICENSE](LICENSE).
