import pyomo.environ as pyo
from neuralprophet import NeuralProphet
import pandas as pd

""" Refactored Code from actual project """

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
        self.model = NeuralProphet(n_changepoints=10, epochs=50) # arbitrary parameters

    def train_forecaster(self):
        self.model.fit(self.history_df, freq='H', silent=True)

    def forecast_generator(self, periods):
        """Yield forecasts one at a time without storing."""
        future = self.model.make_future_dataframe(self.history_df, periods=periods, n_historic_predictions=False)
        forecast = self.model.predict(future)[['ds', 'yhat1']].tail(periods)
        for t, (_, row) in enumerate(forecast.iterrows()):
            yield t, float(row['yhat1'])

    def populate_forecasts(self, periods):
        """Populate forecast_dict by consuming the generator."""
        for t, value in self.forecast_generator(periods):
            self.forecast_dict[t] = value

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

# ─── Example Usage ──────────────────────────────────────────
if __name__ == "__main__":
    ds = pd.date_range('2025-10-20', periods=48, freq='H')
    hist_df = pd.DataFrame({'ds': ds, 'y': [5 + (i % 6) * 0.5 for i in range(len(ds))]})

    nc = NonControllableModule(name="BaseLoad", power_kW=250, history_df=hist_df)
    nc.train_forecaster()
    nc.populate_forecasts(periods=24)

    ev1 = ControllableModule(name="EV1", power_kW=12, soc=35, soc_full=67)
    ev1.soc_update(1)
    print("EV1 SoC:", ev1.soc)