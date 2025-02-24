# =================== PLOTTING =================== #

def save_plot(fig, name):
    fig.savefig(f"{graph_dir}/{name}", bbox_inches='tight', dpi=300)
    plt.close(fig)

# 1️⃣ Transactions Per Account
fig, ax = plt.subplots(figsize=(10, 5))
account_counts = df_tx['from'].value_counts()
colors = coolwarm_colors(np.linspace(0, 1, len(account_counts)))
bars = account_counts.plot(kind='bar', color=colors, ax=ax)
ax.set_xticks(range(len(account_counts)))
ax.set_xticklabels(range(1, len(account_counts) + 1), rotation=45, ha='right')
total_transactions = len(df_tx)
ax.text(0.9, 0.9, f"Total Transactions: {total_transactions}", 
        transform=ax.transAxes, fontsize=12, color='black', 
        bbox=dict(facecolor='white', alpha=0.7))
ax.set_xlabel('Account Number')
ax.set_ylabel('Number of Transactions')
ax.set_title('Transactions Per Account')
ax.grid(axis='y', linestyle='--', alpha=0.6)
save_plot(fig, "transactions_per_account.png")

# 2️⃣ Transaction Frequency Over Time
df_tx.set_index(pd.to_timedelta(df_tx['relative_time'], unit='s'), inplace=True)
tx_time_series = df_tx.resample('1T').size()
fig, ax = plt.subplots(figsize=(10, 5))
tx_time_series.plot(kind='line', color=coolwarm_colors(5), marker='o', ax=ax)
total_time = df_tx.index.max()
ax.text(0.9, 0.9, f"Total Time: {total_time.total_seconds():.1f}s", 
        transform=ax.transAxes, fontsize=12, color='black', 
        bbox=dict(facecolor='white', alpha=0.7))
ax.set_xlabel('Time (seconds from start)')
ax.set_ylabel('Number of Transactions')
ax.set_title('Transaction Frequency Over Time')
ax.grid(True, linestyle='--', alpha=0.6)
save_plot(fig, "transaction_frequency.png")

# 3️⃣ Gas Usage Distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_tx['gas_used'], bins=20, color=coolwarm_colors(7), alpha=0.7, edgecolor='black')
total_gas_used = df_tx['gas_used'].sum()
ax.text(0.9, 0.9, f"Total Gas Used: {total_gas_used}", transform=ax.transAxes, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))
ax.set_xlabel('Gas Used')
ax.set_ylabel('Frequency')
ax.set_title('Gas Usage Per Transaction')
ax.grid(axis='y', linestyle='--', alpha=0.6)
save_plot(fig, "gas_usage.png")

# 4️⃣ Transaction Latency Analysis
df_tx['latency'] = df_tx.index.to_series().diff().dt.total_seconds()
fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(df_tx.index, df_tx['latency'], c=df_tx['latency'], cmap='coolwarm', alpha=0.7, edgecolors='black')
avg_latency = df_tx['latency'].mean()
ax.text(0.9, 0.9, f"Avg Latency: {avg_latency:.2f}s", transform=ax.transAxes, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))
ax.set_xlabel('Time (seconds from start)')
ax.set_ylabel('Transaction Latency (Seconds)')
ax.set_title('Transaction Latency Over Time')
fig.colorbar(sc, label="Latency (s)")
ax.grid(True, linestyle='--', alpha=0.6)
save_plot(fig, "transaction_latency.png")

# 5️⃣ Success vs. Failure Rate
total_rows = 100
success = len(df_tx)
failures = total_rows - success
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([success, failures], labels=['Success', 'Failure'], autopct='%1.1f%%', colors=[coolwarm_colors(3), coolwarm_colors(9)], startangle=90)
ax.set_title('Transaction Success vs. Failure Rate')
save_plot(fig, "success_vs_failure.png")

# 6️⃣ Blockchain vs. Traditional Database Storage
blockchain_size = len(df_tx) * df_tx.memory_usage().sum() / 1024
traditional_db_size = total_rows * 50
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([blockchain_size, traditional_db_size], labels=['Blockchain Storage', 'Traditional DB Storage'], autopct='%1.1f%%', colors=[coolwarm_colors(4), coolwarm_colors(8)], startangle=90)
ax.set_title('Blockchain vs. Traditional Database Storage')
ax.text(0.9, 0.9, f"Blockchain Storage: {blockchain_size:.2f} KB", transform=ax.transAxes, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))
save_plot(fig, "blockchain_vs_db.png")

print("✅ All graphs generated successfully with total values!")
