import pyomo.environ as pyo
from neuralprophet import NeuralProphet
import pandas as pd
from datetime import datetime, timedelta

# ─── Base Module ─────────────────────────────────────────────
class BaseModule:
    def __init__(self, name, power_kW):
        self.name = name
        self.power = power_kW

    def get_power(self, t):
        raise NotImplementedError


# ─── Non-Controllable Module ────────────────────────────────
class NonControllableModule(BaseModule):
    def __init__(self, name, power_kW, history_df):
        super().__init__(name, power_kW)
        self.history_df = history_df
        self.forecast_dict = {}
        self.model = NeuralProphet(n_changepoints=10, epochs=50)

    def train_forecaster(self):
        self.model.fit(self.history_df, freq='H', silent=True)

    def forecast_generator(self, periods):
        """Yield forecasted values hour by hour."""
        future = self.model.make_future_dataframe(self.history_df, periods=periods, n_historic_predictions=False)
        forecast = self.model.predict(future)[['ds', 'yhat1']].tail(periods)
        for t, (_, row) in enumerate(forecast.iterrows()):
            value = float(row['yhat1'])
            self.forecast_dict[t] = value
            yield t, value

    def get_power(self, t):
        return self.forecast_dict.get(t, 0.0)


# ─── Controllable Module ────────────────────────────────────
class ControllableModule(BaseModule):
    def __init__(self, name, power_kW, soc, soc_full):
        super().__init__(name, power_kW)
        self.soc = soc
        self.soc_full = soc_full

    def soc_update(self, decision):
        self.soc = min(self.soc_full, self.soc + decision * self.power)
        return self.soc

    def get_power(self, t):
        return self.power


# ─── Optimization Manager ───────────────────────────────────
class ModuleManager:
    def __init__(self, controllable_modules, nc_modules, granularity=1):
        self.ctrl_modules = controllable_modules
        self.nc_modules = nc_modules
        self.granularity = granularity
        self.last_step = 24 * granularity
        # Precompute static maps for faster constraint expressions
        self.name_to_power = {mod.name: mod.power for mod in self.ctrl_modules}

    def controllable_modules(self):
        for m in self.ctrl_modules:
            yield m

    def build_model(self, elec_prices):
        m = pyo.ConcreteModel()
        m.time = pyo.RangeSet(0, self.last_step - 1)
        m.modules = pyo.Set(initialize=[mod.name for mod in self.ctrl_modules])
        m.u = pyo.Var(m.time, m.modules, domain=pyo.Binary)

        # Aggregate all non-controllable forecasts
        nc_forecast = {t: sum(nc.get_power(t) for nc in self.nc_modules)
                       for t in range(self.last_step)}

        def obj_expr(mdl):
            return sum(elec_prices[t] * sum(mdl.u[t, i] for i in mdl.modules) for t in mdl.time)
        m.obj = pyo.Objective(rule=obj_expr)

        def demand_rule(mdl, t):
            nc = nc_forecast[t]
            # Ensure controllable generation meets the non-controllable demand
            return sum(mdl.u[t, i] * self.name_to_power[i] for i in mdl.modules) >= nc
        m.demand_constraint = pyo.Constraint(m.time, rule=demand_rule)
        self.model = m
        return m

    def solve_model(self, solver='highs', time_limit=60):
        opt = pyo.SolverFactory(solver)
        opt.options['time_limit'] = time_limit
        res = opt.solve(self.model, tee=False)

        if (res.solver.status == pyo.SolverStatus.ok) and \
           (res.solver.termination_condition == pyo.TerminationCondition.optimal):
            decisions = {mod.name: [] for mod in self.ctrl_modules}
            for t in self.model.time:
                for mod in self.ctrl_modules:
                    dec = int(round(self.model.u[t, mod.name].value))
                    decisions[mod.name].append(dec)
                # Sequentially update SoC after each time step
                for mod in self.ctrl_modules:
                    mod.soc_update(decisions[mod.name][-1])
            return decisions
        else:
            raise RuntimeError(f"Solver failed: {res.solver.termination_condition}")


# ─── Example Usage ──────────────────────────────────────────
if __name__ == "__main__":
    # Non-controllable historical data
    ds = pd.date_range('2025-10-20', periods=48, freq='H')
    hist_df = pd.DataFrame({'ds': ds, 'y': [5 + (i % 6) * 0.5 for i in range(len(ds))]})

    nc = NonControllableModule("BaseLoad", power_kW=5, history_df=hist_df)
    nc.train_forecaster()
    for t, f in nc.forecast_generator(periods=24):
        pass  # yields forecasted demand for each hour

    # Controllable modules
    ev1 = ControllableModule("EV1", 7, soc=40, soc_full=100)
    ev2 = ControllableModule("EV2", 10, soc=60, soc_full=100)

    # Build optimization manager
    mgr = ModuleManager([ev1, ev2], [nc], granularity=1)
    prices = {t: 0.1 + 0.02 * (t % 6) for t in range(24)}
    model = mgr.build_model(prices)
    decisions = mgr.solve_model()
    print("Decisions t=0:", decisions)
    print("Updated SoC EV1:", ev1.soc)
    print("Updated SoC EV2:", ev2.soc)
