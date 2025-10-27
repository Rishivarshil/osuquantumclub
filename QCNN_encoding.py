import numpy as np
import pandas as pd
import math
import yfinance as yf # type: ignore
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp

def fetch_close_prices(ticker="^GSPC", start="2022-01-01", end=None):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker or date range.")

    # Handle inconsistent column naming
    if "Adj Close" in data.columns:
        close = data["Adj Close"]
    elif "Close" in data.columns:
        close = data["Close"]
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in yfinance data columns.")
    
    return close


def preprocess_prices_to_returns(close_series, window_ma=None):
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    returns = close_series.pct_change().dropna()
    if window_ma and window_ma > 1:
        returns = returns.rolling(window=window_ma, min_periods=1).mean()
    return returns.squeeze()


def quantize_series(returns_series, d=2, clip_percentile=(0.5, 99.5)):
    vals = np.array(returns_series).flatten()

    lowp, highp = np.percentile(vals, clip_percentile)
    vals_clipped = np.clip(vals, lowp, highp)
    xmin, xmax = vals_clipped.min(), vals_clipped.max()

    bins = np.linspace(xmin, xmax, num=d+1)
    labels = np.digitize(vals_clipped, bins[1:-1], right=False)
    labels = np.clip(labels, 0, d-1)

    return pd.Series(labels, index=returns_series.index), bins


def build_context_windows(label_series, T):
    contexts = []
    labels = label_series.values
    idx = label_series.index
    for i in range(T - 1, len(labels)):
        ctx = tuple(labels[i - T + 1 : i + 1])
        contexts.append((idx[i], ctx))
    return contexts


def calculate_context_distribution(contexts, T, d):
    keys = [sum(c * (d ** (T - 1 - i)) for i, c in enumerate(ctx)) for _, ctx in contexts]

    counts = pd.Series(keys).value_counts().sort_index()
    
    distribution = counts / len(keys)
    
    target_dim = d**T
    target_dist = np.zeros(target_dim)
    for k, v in distribution.items():
        if k < target_dim:
            target_dist[k] = v
            
    target_amplitudes = np.sqrt(target_dist)
    
    return target_amplitudes, target_dist


def hardware_efficient_ansatz(params, n_qubits, n_layers):

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[2 * i + 0 + 2 * n_qubits * l], wires=i)
            qml.RZ(params[2 * i + 1 + 2 * n_qubits * l], wires=i)

        for j in range(n_qubits):
            qml.CNOT(wires=[j, (j + 1) % n_qubits])
            
    for j in range(n_qubits):
        qml.CNOT(wires=[j, (j + 1) % n_qubits])
        
dev_sp = qml.device("default.qubit", wires=3)

@qml.qnode(dev_sp)
def state_preparation_circuit(params, n_qubits, n_layers):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        
    hardware_efficient_ansatz(params, n_qubits, n_layers)
    
    return qml.state()


def plot_price_series(close):
    plt.figure(figsize=(10,4))
    plt.plot(close.index, close.values, label="S&P 500 (Close)")
    plt.title("S&P 500 Price History")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_returns_distribution(returns):
    plt.figure(figsize=(7,4))
    plt.hist(returns, bins=40, color='steelblue', alpha=0.7)
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_quantization_bins(returns, bins):
    plt.figure(figsize=(8,4))
    plt.hist(returns, bins=40, color='lightgray', edgecolor='k', alpha=0.6)
    for b in bins:
        plt.axvline(b, color='red', linestyle='--', alpha=0.6)
    plt.title("Quantization Bins for Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    T = 3       
    d = 2       
    start = "2023-01-01"

    close = fetch_close_prices(start=start)
    plot_price_series(close)

    returns = preprocess_prices_to_returns(close)
    plot_returns_distribution(returns)

    labels, bins = quantize_series(returns, d=d)
    plot_quantization_bins(returns, bins)

    contexts = build_context_windows(labels, T)
    target_amplitudes, target_probabilities = calculate_context_distribution(contexts, T, d)
    N_qubits_T = T
    
    print(f"\nTarget Probability Distribution P(X^({T})) (Size {d**T}):")
    print(target_probabilities[:1])
    print(f"Total probability sum: {np.sum(target_probabilities):.4f}")


