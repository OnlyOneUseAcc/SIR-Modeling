import pandas as pd
from scipy.optimize import minimize
import numpy as np
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Learner(object):
    def __init__(self, country, start_date, N):
        self.start_date = start_date
        self.N = N
        self.country = country

    def load_confirmed(self):
        df = pd.read_csv('data/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:'6/19/21']

    def load_immune(self):
        df = pd.read_csv('data/time_series_19-covid-Recovered-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:'6/19/21']

    def load_dead(self):
        df = pd.read_csv('data/time_series_19-covid-Deaths-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:'6/19/21']

    def predict(self):
        size = len(self.immunne) - self.train_index

        beta = self.beta
        gamma = self.gamma

        def d_sir(y, t):
            S = y[0]
            I = y[1]
            R = y[2]
            y0 = -beta * S * I
            y1 = beta * S * I - gamma * I
            y2 = gamma * I

            return [y0, y1, y2]

        s_0 = (self.N - (self.confirmed[self.train_index] + self.immunne[self.train_index])) / self.N
        i_0 = self.confirmed[self.train_index]
        r_0 = self.immunne[self.train_index]
        y0 = [s_0, i_0, r_0]
        tspan = np.arange(0, size, 1)
        res = odeint(d_sir, y0, tspan)

        y0 = res[:, 0]
        y1 = res[:, 1] * 1.5
        y2 = res[:, 2] * 1.5

        return y0, y1, y2

    def train(self, split_value=0.7):
        self.healed = self.load_immune()
        self.death = self.load_dead()

        self.immunne = (self.healed + self.death) / self.N # R
        self.confirmed = self.load_confirmed() / self.N # I
        self.potencial = (self.N - (self.confirmed + self.immunne)) / self.N

        self.train_index = int(self.confirmed.shape[0] * split_value)

        self.s_0 = (self.N - (self.confirmed[0] + self.immunne[0])) / self.N
        self.i_0 = self.confirmed[0] / self.N
        self.r_0 = self.immunne[0] / self.N

        optimal = minimize(self.loss, np.array([0.01, 0.01]),
                           args=(self.confirmed.iloc[:self.train_index],
                                 self.immunne.iloc[:self.train_index],
                                 (self.N - (self.confirmed[0] + self.immunne[0])) / self.N,
                                 self.confirmed[0] / self.N,
                                 self.immunne[0] / self.N),
                           method='L-BFGS-B',
                           bounds=[(0.00000001, 1.0), (0.00000001, 1.0)])
        print(optimal)
        beta, gamma = optimal.x
        self.beta = beta
        self.gamma = gamma

    def derivative_sir(self, y, t):
        S = y[0]
        I = y[1]
        R = y[2]
        y0 = -self.beta * S * I
        y1 = self.beta * S * I - self.gamma * I
        y2 = self.gamma * I

        return [y0, y1, y2]

    def loss(self, point, potencial, immune, s_0, i_0, r_0):
        size = len(potencial)
        beta, gamma = point

        def SIR(y, t):
            S = y[0]
            I = y[1]
            R = y[2]
            y0 = -beta * S * I
            y1 = beta * S * I - gamma * I
            y2 = gamma * I
            return [y0, y1, y2]

        y0 = [s_0, i_0, r_0]
        tspan = np.arange(0, size, 1)
        res = odeint(SIR, y0, tspan)
        l1 = np.sqrt(np.mean((res[:, 0] - potencial) ** 2))
        l2 = np.sqrt(np.mean((res[:, 2] - immune) ** 2))

        return l1 + l2

