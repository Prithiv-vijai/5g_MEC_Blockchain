docker build -t lightgbm-container .

docker-compose up


eMBB -  Background_Download, File_Download, Streaming, Video_Streaming
URLLC - Emergency_Service, Online_Gaming, Video_Call, VoIP_Call, Voice_Call
mMTC - IoT_Temperature, Web_Browsing


# Updated Signal Strength Calculation (Inverse of Distance Formula)
def calculate_signal_strength(d):
    if d > 0:
        return Pt - PL_d0 - 10 * n * np.log10(d / d0)
    return np.nan

# Updated Latency Calculation (Inverse of Distance Formula)
def calculate_latency(d):
    if d > 0:
        return (2 * d) / (c * k) * 1000  # Latency in ms
    return np.nan

# Apply the reverse calculations
df['Updated_Signal_Strength'] = df.apply(
    lambda row: calculate_signal_strength(row['New_Distance']) if row['New_Distance'] > 0 else row['Signal_Strength'],
    axis=1
)

df['Updated_Latency'] = df.apply(
    lambda row: calculate_latency(row['New_Distance']) if row['New_Distance'] > 0 else row['Latency'],
    axis=1
)