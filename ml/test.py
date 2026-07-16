    fig, ((predictions, precipitation), (temperature, wind_speed)) = plt.subplots(2, 2, figsize =(13,6))

    current_time = datetime.now(UTC).replace(
        minute=0,
        second=0,
        microsecond=0
    )

    x_text = (current_time + timedelta(minutes=45)).isoformat()
    x_border = (current_time - timedelta(minutes = 60)).isoformat()

    current_time = current_time.isoformat()

    current_time = pd.to_datetime(current_time)
    x_text = pd.to_datetime(x_text)
    x_border = pd.to_datetime(x_border)

    before = df[df['measuredAt'] <= current_time]

    # after contains instances before the current time, therefore it is good to add the vertical line and indicator
    after = df[df['measuredAt'] > x_border]

    level = df.loc[df['measuredAt'] == current_time, 'levelAtHour'].iloc[0]
    precip = df.loc[df['measuredAt'] == current_time, 'precipitation'].iloc[0]
    temp = df.loc[df['measuredAt'] == current_time, 'temperature_2m'].iloc[0]
    wind = df.loc[df['measuredAt'] == current_time, 'wind_speed_10m'].iloc[0]

    # up to this point, we have been using utc curren time for all operations

    # used to be ax -> before_predictions
    before_predictions = before.plot(
        kind = 'line', 
        x='measuredAt', 
        y='levelAtHour', 
        color='#0057E7',
        ax = predictions
    )

    # used to be ax2 -> after_predictions
    after_predictions = after.plot(
        kind = 'line', 
        x='measuredAt', 
        y='levelAtHour', 
        color='#0057E7',
        linestyle="--", 
        ax = before_predictions
    )


    before_precipitation = before[['precipitation','measuredAt']].plot(
        kind = "line",
        x = "measuredAt",
        y = "precipitation",
        color='#17becf',
        ax = precipitation
    )

    after_precipitation = after[['precipitation','measuredAt']].plot(
        kind = 'line',
        linestyle='--',
        x='measuredAt',
        y='precipitation',
        color='#17becf',
        ax = before_precipitation
    )

    before_temperature = before[['temperature_2m','measuredAt']].plot(
        kind = "line",
        x = "measuredAt",
        y = "temperature_2m",
        color='#ff7f0e',
        ax = temperature
    )

    after_temperature = after[['temperature_2m','measuredAt']].plot(
        kind = 'line',
        linestyle='--',
        x='measuredAt',
        y='temperature_2m',
        color='#ff7f0e',
        ax = before_temperature
    )

    before_wind_speed = before[['wind_speed_10m','measuredAt']].plot(
        kind = "line",
        x = "measuredAt",
        y = "wind_speed_10m",
        color='#2ca02c',
        ax = wind_speed
    )

    after_wind_speed = after[['wind_speed_10m','measuredAt']].plot(
        kind = 'line',
        linestyle='--',
        x='measuredAt',
        y='wind_speed_10m',
        color='#2ca02c',
        ax = before_wind_speed
    )


    
    for ax in (predictions, precipitation, temperature, wind_speed):
        ax.get_legend().remove()
        ax.spines["top"].set_linewidth(0)
        ax.spines["bottom"].set_color("gray")
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_color("gray")
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(0)
        ax.set_xlabel("Time (Toronto/America)")

    after_predictions.set_ylabel("Water Level (m)")
    after_precipitation.set_ylabel("Precipitation (mm)")
    after_temperature.set_ylabel("Temperature (C)")
    after_wind_speed.set_ylabel("Wind Speed (C)")

    after_predictions.scatter(
        current_time,
        level,
        color = '#0057E7',
        marker = 'o',
        s=30
    )
    after_precipitation.scatter(
        current_time,
        precip,
        color = "#0057E7",
        marker = 'o',
        s = 30
    )
    after_temperature.scatter(
        current_time,
        temp,
        color = "#0057E7",
        marker = 'o',
        s = 30
    )
    after_wind_speed.scatter(
        current_time,
        wind,
        color = "#0057E7",
        marker = 'o',
        s = 30
    )


    after_predictions.text(
        x = x_text,
        y = level + 0.001,
        alpha = 1.0,
        s = str(round(level,2)) + "m", # get 0th indexed value from series pandas
        fontsize=10,
        color="gray",
        fontweight = 'bold'
    )
    
    after_precipitation.text(
        x = x_text,
        y = precip + 0.001,
        alpha = 1.0,
        s = str(round(precip,2)) + "mm", # get 0th indexed value from series pandas
        fontsize=10,
        color="gray",
        fontweight = 'bold'
    )
    
    after_temperature.text(
        x = x_text,
        y = temp + 0.001,
        alpha = 1.0,
        s = str(round(temp,2)) + "C", # get 0th indexed value from series pandas
        fontsize=10,
        color="gray",
        fontweight = 'bold'
    )

    after_wind_speed.text(
        x = x_text,
        y = wind + 0.001,
        alpha = 1.0,
        s = str(round(wind,2)) + "km/h", # get 0th indexed value from series pandas
        fontsize=10,
        color="gray",
        fontweight = 'bold'
    )



    for ax in (after_predictions, after_precipitation, after_temperature, after_wind_speed):
        ax.axvline(
            x = current_time,
            color = 'gray',
            linewidth = 1,
            linestyle = "--"
        )

        timesAfter = map(getTimes, ax.get_xticklabels())

        positions = ax.get_xticks()
    
        ax.set_xticks(positions)

        ax.set_xticklabels(timesAfter, rotation=0, ha='center')

    predictions.set_title("Predictions")
    precipitation.set_title("Precipitation")
    temperature.set_title("Temperature")
    wind_speed.set_title("Wind Speed")

    fig.subplots_adjust(
        hspace=0.5
    )
    # dot for current level/current rain text
    # current level/current rain text
    # ylabel
