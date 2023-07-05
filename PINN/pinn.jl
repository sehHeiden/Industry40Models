### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 32ac2420-0ad8-11ee-2d6f-2d8c15a32168
begin
	using CSV
	using DataFrames
	using StatsPlots
	using NLsolve
	using Flux
	using JLD2
	using RollingFunctions
	using PlutoUI
	using StatsBase
	using Random
end

# ╔═╡ d8760ed7-947f-4a3c-be31-08f67020e769
md"""
# Physically Informed Neural Network (PINN)
"""

# ╔═╡ 8733e584-78c6-41c4-9f30-e86ada17fe8f
TableOfContents()

# ╔═╡ 993eda49-5f21-4368-94a6-4801f949a97b
md"""
##  Laden und Aufbereitung der Messdaten.
"""

# ╔═╡ ed1920f6-e175-4032-9f8b-eb26a49ea193
celsius2kelvin(c) = c + 273.15

# ╔═╡ bf2bea30-98a5-47f4-aa2b-108e88eac174
function prepare_dataframe(data_frame_path)

    _df = DataFrame(CSV.File(data_frame_path));
    start_times_stat = unique(_df, :run);

    _df.start_time =  start_times_stat[trunc.(Int, _df.run).+1, :time];
    _df.duration = _df.time - _df.start_time;
	
    _df.T1 = _df.T1 .|> celsius2kelvin;
	_df.T2 = _df.T2 .|> celsius2kelvin;
	
    _df.Column1 = _df.Column1 .+ 1;
	_df.run = _df.run .+ 1;
	_df.global_duration  = _df.time .- _df.time[1];
	
	_df.dT1 = prepend!([0.0, ], diff(_df.T1))
	_df.dT2 = prepend!([0.0, ], diff(_df.T2))
	_df.dt = prepend!([0.0, ], diff(_df.time))

	_df.dT1dt = _df.dT1  ./ _df.dt
	_df.dT2dt = _df.dT2  ./ _df.dt

	# fill the delta for the prepended value
	replace!(_df.dT1dt, NaN => 0)  
	replace!(_df.dT2dt, NaN => 0)
	
    return _df;
end

# ╔═╡ 12487f1f-4209-443f-945a-8c85a0191aac
begin
	heating_df_path = joinpath(@__DIR__, "measurements_heating_and_cooling.csv");
	heating_df = prepare_dataframe(heating_df_path);
	heating_curves  =  heating_df |> Matrix;
end

# ╔═╡ 95e49920-3c3b-4467-914a-9a93714cc9de
heating_df

# ╔═╡ 0dd3c3d4-3758-4430-a403-96b49a42d44c
md"""
Unten wird der vollständige Verlauf der Heizkurven dargestellt. In der Darstellung sind die Testpunkte für den physikalischen Loss enthalten.

Für den physikalischen Loss werden max(30, Batchgröße) Messpunkte aus dem ersten bis fünften Lauf zufällig ausgewählt. Aktuell wird eine Batchgröße von 64 gewählt. Die Wahrscheinlichkeit der Auswahl ist proprotional dem Quadrat des Differenzenquotienten des rollenden Mittelwerts mit Fensterbreite 10 und dem Quadrat des Abstandes der Messtemperatur von der mittleren Temperatur. Hintergrund ist, dass die Netzwerke erst bei der Optimierung die mittere Temperatur finden und dann viel langsamer sich an den Kurvenverlauf anpassen. Um dies zu Beschleuigen werden Tupel aus (Zeit,Temperatur) gewählt die von der Mittelwerttemperatur abweichen, sodass der physikalische Loss gleichzeitig als Strafterm für die Optimierung hin zur Mittelwerttemperatur genutzt wird. 
Trainiert wird auf der ersten Hälfte des Datensatzes.
Da der Loss für jeden Batch bestimmt wird, werden pro Batch der physikalische Loss, und der datengestüzte Loss berechnet. Daher ist ein Anpassungsfaktor des datengestützen Loss und des physikalischen Loss abhängig von der Batchgröße nötig. Deswegen ist die Größe des Testdatensatzes für den physikalischen Loss abhägnig von der Batchgröße. Online eine Feinanpassung zu ermöglichen kann nützlich sein.
Es wurde ein Korrekturfaktor in die physikalischen Lossfunktionen eingebaut, der Erfahrungsgemäß die **Startwerte**  der physkalischen Lossfunktionen der MSE anpasst. Wegen zufälliger Initialisierung der Layerparameter, kann es zu leichten Abweichungen kommen.
"""

# ╔═╡ 1a2337f0-bede-4d7d-8fdb-125406dbb703
md"""
## Default Parameter
"""

# ╔═╡ 90fa8c5e-e048-4133-b8d3-89116bab7ae4
begin
	epochs = 10_000
	η = 1e-4;

	train_mask = 1: Int.(length(heating_df.T1)//2);

	x_full_df = heating_df[:,[:global_duration]];
	x_full_m = x_full_df |> Matrix;
	x_full_m  =  [x for x in eachrow(x_full_m)];
end

# ╔═╡ cf7f7ede-39d8-4ab3-b604-6b076e14c978
begin
	batchsize = 64
	train_df = heating_df[train_mask, :];
	x_train_m = train_df[:,[:global_duration]] |> Matrix;

	y_train = train_df[:, [:T1]] |> Matrix

	loader = Flux.DataLoader((data=x_train_m', label=y_train'), batchsize=batchsize, shuffle=true);
end

# ╔═╡ 6a9e72c0-60de-44b4-ab2a-dcf4c6bd2138
begin
	p1 = @df heating_df plot(:global_duration, [:T1 ],
		group = Int.(heating_df.run), 
		legend = :topright,
		xlabel = "time ∖s", 
		ylabel = "T  ∖°C",
		title = "Heating Curves", 
		lw = 4, 
		fmt = :png,
		palette=cgrad(:viridis, rev=true, categorical = true))

	t =  heating_df.run .∈ Ref([1, 2, 3, 4, 5])
	dT₁ = heating_df.T1[t] |> (y -> runmean(y, 10)) |> diff |> (y -> prepend!([0.0, ], y))
	rowsΔ = StatsBase.sample(heating_df.Column1[t], Weights(abs2.(dT₁) .* abs2.(heating_df.T1[t] .- mean(heating_df.T1[t]))), maximum([30, batchsize]))
	T1Δ = heating_df.T1[rowsΔ]
	tΔ = heating_df.global_duration[rowsΔ]

	scatter!(p1, tΔ, T1Δ, label="phy. Loss Samples", mc=:red);


end

# ╔═╡ 9d979f43-e5bb-47c9-90d2-e6194483e8f3
md"""
Die Netze sind zwar durch mit rein zufälligen Werten initialisiert. Beginnen dabei aber dennoch mit den selben Startparametern, da Parameter für das PINN mittels DeepCopy aus dem Netz für das klassische MLP kopiert wurden.
"""

# ╔═╡ c4d1c1d8-8fa8-441c-9713-9560c12ad2e9
begin
	NN = Chain(
		Dense(1 => 32, tanh),
		Dense(32 => 32, tanh),
		Dense(32 => 1)
	) |> f64;

	NN_transfer = deepcopy(NN);
	NN_fopdt = deepcopy(NN);

end

# ╔═╡ 757e10e7-1cd2-476c-a189-923aad028ae9
md"""
## Classical MLP
"""

# ╔═╡ c1277d84-5430-4682-a751-b009f4df87d0
begin
	opt_nn = Flux.setup(Adam(η), NN)
	losses_nn = []
	
	for i ∈ 1:epochs
		local_loss = []
		for (x, y) in loader
			loss, grads = Flux.withgradient(NN) do m
				ŷ = m(x)
				Flux.mse(ŷ, y)
			end
			Flux.update!(opt_nn, NN, grads[1])
			push!(local_loss, loss)
		end
		push!(losses_nn, Flux.mean(local_loss))
	end
end

# ╔═╡ 2824ad6b-b032-42c8-927c-ce6308c7d7bc
Tpred_nn = [NN(x) for x in x_full_m] .|> first;

# ╔═╡ 0f22424a-6147-4326-bed4-65061502d496
plot(1:epochs, log10.(losses_nn), 
	title="Loss",
	label="MSE MLP",
	xlabel="Epoch", 
	ylabel="log₁₀ mse",)

# ╔═╡ 35aed450-88a5-460a-92b0-39375a28a165
plot(heating_df.global_duration, 
	[heating_df.T1 Tpred_nn],
	label=["Heizkurve" "Vorhersage MLP"],
	xlabel="t /s", 
	title="Vollständige Heizkurven",
	ylabel="T /K",)

# ╔═╡ 2d6f1e50-4bfa-47fa-8ae2-728506db9b19
md"""
Beim Vergleich von Ergebnissen nach unterschiedlichern Anzahl von Epochen zeigt sich, dass die Optimierung mit der Vorhergesage die mittlere Temperatur während der Messung beginnt. Danach wird die Temperatur für jeden Lauf nacheinander optimiert.

### Speichern des Modelles

"""

# ╔═╡ fdf54f26-8169-405e-a288-8247ed275c99
begin
	nn_state = Flux.state(NN);
	jldsave("nn_model.jld2"; nn_state)
end

# ╔═╡ a372911c-2c18-495e-9098-4bc76a6700ed
md"""

## PINN - FOPDT
### Fitten des FOPDT

Der physikalische Loss-Termin wird mittels [first order plus dead time (FOPDT)-Modell](https://apmonitor.com/pdc/index.php/Main/FirstOrderOptimization) erstellt, um die Faktoren τ ,Κ, θ zu bestimmen. 
Da τ in die Abkühlung eingeh, Κ und θ hingegen nicht, wird der Term erst für die Abkühlung berechnet. Danach wird der Faktor τ in die Heizungsphase eingesetzt.

Das ODE wird mit nlsove gelöst.
Eingesetzt werden die Messwerte den ersten, 75ten und 130ten Messpunkt. 

"""

# ╔═╡ 9d9ef719-b19b-471e-b6ba-3bcf07fded6f
function FOPDT_off(z)
    τ = z[1];
	
    return [dT + celsius2kelvin(T) / τ for (dT, T) in zip([-0.2425,], [76.006,])];
end

# ╔═╡ 2683688e-e96d-42b3-80cb-ae12ec4d767d
begin
	function FOPDT_on(z)
	    τ = 1439.8;
		Κ, θ = z;
		
	    return [dT + (1 / τ) * (celsius2kelvin(T) + Κ * 1 *(θ -i)) for (dT, T, i) in zip(
			[0.279, 0.3065],
			[46.035,60.279], 
			[75, 130])];
	
	end
end

# ╔═╡ d8e34379-6d4b-4149-a2f4-a4944a28f2c4
begin
	initial_x_off = [1439.8, ];
	sol_off = nlsolve(FOPDT_off, initial_x_off, xtol=1e-7, ftol=1e-6);
	sol_off
end

# ╔═╡ bfb4ace1-1c7f-4305-b83a-f6c7a6feba3f
begin
	initial_x_on = [0.0097889, -666.1];
	sol_on = nlsolve(FOPDT_on, initial_x_on, xtol=1e-6, ftol=1e-6);
	sol_on
end

# ╔═╡ d4b240f7-cbf9-4fa2-9046-bf38661b1d9c
md"""
### Erstellen des Beispieldatensatzes für die Physikalischen Lossfunktionen
"""

# ╔═╡ 5e1c95cc-46e3-4161-aaa1-0232ee06b119
begin
	is_at_sample_time = heating_df.Column1 .∈ Ref(rowsΔ);
	sample_df = heating_df[is_at_sample_time, :];
	samples = sample_df |> Matrix;
end

# ╔═╡ cc3cd8e5-5749-4165-b9ee-e3199a0151b3
md"""
### Erstellen des FOPDT-Losses
Das physikalische Model wird mittels FOPDT ertellt. Die bereits ermittelten Faktoren Κ, τ, θ werden hier eingesetzt. 
Die aktuelle Temperatur wurd aus dem neuronalen Netzwerk vorhergesagt. Daher wird die linke Seite des ODE durch das Delta der Vorhersagetemperatur um die Beispielsmesspunkte berechnet. Beide Seiten des ODE müssen gleich sein, daher wird diese Bedingung zum Minimierung (als Loss-Term, oder [Regularisierung](https://fluxml.ai/Flux.jl/stable/training/training/#Regularisation)) genutzt. 
"""

# ╔═╡ 46ae8dc5-e139-4018-97ef-03581ccf6e26
begin

	function loss_fopdt(x)

		Κ = 0.9788818181818202;
		τ = 1439.818556461867;
		θ = -661.4415056140165;

		# t_global_duration := Spalte 9 im Datensatz
		ŷₙₑₓₜ = [x[9] + 0.5] |> NN_fopdt |> first;
		ŷₚᵣₑᵥ = [x[9] - 0.5] |> NN_fopdt |> first;
		Δy_pred = ŷₙₑₓₜ - ŷₚᵣₑᵥ;

		ŷ = [x[9]] |> NN_fopdt |> first;
		U = x[4] /100;
		i₀ = x[8];  # Zeit seit letzten Start der Heizung
		
		yᵣ = (.-ŷ .+ Κ .* U .*(i₀ .- θ)) ./ τ;
		
		return sum((Δy_pred .- yᵣ) .^2)
	end
	loss_fopdt() = eachrow(samples) .|> loss_fopdt |> sum |> (x)->(x*13_000)
end

# ╔═╡ 99671fef-cd7b-4a60-80f8-8c28ffa173f2
md"""
### Trainieren des FOPDT-Modells
"""

# ╔═╡ 96429da4-48a2-4b1d-97d1-7dd3ab6d361c
begin
	opt_fopdt = Flux.setup(Adam(η), NN_fopdt);
	losses_fopdt_mse = [];
	losses_fopdt_phy = [];

	
	for i ∈ 1:epochs
		local_loss_mse = []
		local_loss_phy = []
		for (x, y) in loader
			grads = Flux.gradient(NN_fopdt) do m
				ŷ = m(x);
				(Flux.mse(ŷ, y) + loss_fopdt())/2
			end
			Flux.update!(opt_fopdt, NN_fopdt, grads[1]);
			push!(local_loss_mse, Flux.mse(NN_fopdt(x), y));
			push!(local_loss_phy, loss_fopdt());
		end
		push!(losses_fopdt_mse, Flux.mean(local_loss_mse));
		push!(losses_fopdt_phy, Flux.mean(local_loss_phy));
	end
end

# ╔═╡ f5c5badd-73a4-465d-90dd-a5caa28e8c92
md"""
Als Eingabe in das Netzwerk wird die Zeit seit Start der Messung gewählt.
Batchgröße ist 64. Bei Minibatching musste der physikalische Lossterm skaliert werden.
"""

# ╔═╡ 539dc603-636c-49d9-a402-c2e731d9baee
md"""
Auftragung des über alle Datenpunkte gemittelten Losses.

Oft PINN nicht besser als NN, daher mit Absicht Gewichtung 0.5. 
1 Trainungspunkt in der mae, ist dabei gleich gewichtet der 12 Punkten der physikalischen Lossfunktion.
"""

# ╔═╡ 552252d9-afb3-407f-a168-d2e7ca62f679
begin
	
	plt1 = plot(1:epochs, 
		[log10.(losses_nn) log10.(losses_fopdt_phy) log10.(losses_fopdt_mse)], 
		title = "Loss", 
		lw = [6 2 2], 
		xlabel = "Epoch", 
		ylabel = "Log₁₀ Loss", 
		la=[0.5 1 1], 
		lc=[:red :blue :aqua :cyan],
		label=["NN" "PINN-FOPDT" "PINN-MSE"])
	
	losses_fopdt = convert(Any, (losses_fopdt_phy + losses_fopdt_mse)./2.0)
	plot!(plt1, 1:epochs, log10.(losses_fopdt), lw=2, lc=:darkcyan,label="PINN total")
end

# ╔═╡ 73f00d67-63a6-4429-b613-dc3935861459
Tpred_fopdt = [NN_fopdt(x) for x in x_full_m] .|> first;

# ╔═╡ 011a7d05-935c-4e1c-a200-391d4108c72a
begin
	plot(heating_df.global_duration, [heating_df.T1 Tpred_nn Tpred_fopdt], 
		colour = [:red :blue :green],
		legend = :topright,
		xlabel = "Time ∖s", 
		ylabel = "T  ∖K",
		label = ["Heizkurve" "Vorhersage MLP" "Vorhersage PINN-FOPDT"],
		title = "Heizkurven", 
		lw = [5 2 2])
end

# ╔═╡ ff6a3eb0-ced6-4901-ab72-5e6a95667e5d
md"""
### Speichern des FOPDT-Modells
"""

# ╔═╡ 2bd87b9f-46dc-492f-9074-a6fd11956270
begin
	pinn_fopdt_state = Flux.state(NN_fopdt);
	jldsave("pinn_fopdt_model.jld2"; pinn_fopdt_state)
end

# ╔═╡ 31918ef3-73f2-4f69-9c71-639ba0ecc191
md"""
## PINN - Wärmefluss

Anstelle des angefittenten FOPDT Models wird hier ein mechanistisches Model für das ODE verwendet. Das Model beschreibt das ODE nach dem [Wärmefluss](https://apmonitor.com/pdc/index.php/Main/ArduinoModeling) (heat transfer).

Die notwendigen physikalischen Parameter sind dem obigen Quelle entnommen.

### Erstellen des physikalischen Loss-Terms
"""

# ╔═╡ 9875d167-60c7-4899-a823-41439ffd4fd1
begin
	function loss_transfer(x)
	
		m = 0.004  # kg
		cₚ = 500  # J/(kg K)
		U = 10  # W / (m² K)
		A = 1.2*1e-3  # m2
		T∞ = 294.6 # 296.16 K on the website, taken the starting temperature instead
		ϵ = 0.9
		σ = 5.67*1e-8 # W/(m² K⁴)
		α = 0.01 * x[4]
		T = x[2]
		Q = 1  # W
		
		dTdt = (U * A * (T∞ - T) + ϵ * σ * (T∞^4 - T^4) + α * Q) / (m * cₚ)
	
		t = x[9]  #  time since start
		dTdt_pred =  NN_transfer([t + 0.5]) .- NN_transfer([t - 0.5])
		
		return sum((dTdt_pred .- dTdt) .^2)
	end
	
	loss_transfer() = eachrow(samples) .|> loss_transfer |> sum |> (x) -> (x/10)
end

# ╔═╡ 7563d048-147a-4fbb-8db2-cfee296a31d3
md"""
### Trainieren des Modells
"""

# ╔═╡ 351b6a22-8411-4126-9057-2784c87e3d4f
begin
	opt_transfer = Flux.setup(Adam(η), NN_transfer);
	losses_transfer_mse = [];
	losses_transfer_phy = [];

	
	for i ∈ 1:epochs
		local_loss_mse = []
		local_loss_phy = []
		for (x, y) in loader
			grads = Flux.gradient(NN_transfer) do m
				ŷ = m(x);
				(Flux.mse(ŷ, y) + loss_transfer())/2
			end
			Flux.update!(opt_transfer, NN_transfer, grads[1]);
			push!(local_loss_mse, Flux.mse(NN_transfer(x), y));
			push!(local_loss_phy, loss_transfer());
		end
		push!(losses_transfer_mse, Flux.mean(local_loss_mse));
		push!(losses_transfer_phy, Flux.mean(local_loss_phy));
	end
end

# ╔═╡ 9f52a9c0-6fd4-4d05-97fd-dc80468c7e4f
losses_transfer = (losses_transfer_mse .+ losses_transfer_phy) ./ 2

# ╔═╡ 27e6c2de-a08e-4d24-a8b2-62a9c5c4324b
plot(1:epochs, 
	[log10.(losses_transfer_mse) log10.(losses_transfer_phy) log10.(losses_transfer)], 
	title = "Loss PINN - heat transfer model", 
	lw = [6 2 2], 
	xlabel = "Epoch", 
	ylabel = "Log₁₀ Loss", 
	la=[0.5 1 1], 
	lc=[:red :blue :aqua :cyan],
	label=["PINN-MSE" "PINN-Physical" "PINN-Total"])


# ╔═╡ 015c0f04-85fb-4c2a-82ce-db0ddc848f17
Tpred_transfer = [NN_transfer(x) for x in x_full_m] .|> first;

# ╔═╡ 4d7ac675-5bad-464b-affa-b3096e8342d8
begin
	plot(heating_df.global_duration, [heating_df.T1 Tpred_nn Tpred_fopdt Tpred_transfer], 
		colour = [:red :blue :green :orange],
		legend = :topright,
		xlabel = "Time ∖s", 
		ylabel = "T  ∖K",
		label = ["Heizkurve" "Vorhersage MLP" "Vorhersage PINN-FOPDT" "Vorhersage PINN-Wärmefluss"],
		title = "Vollständige Heizkurven", 
		lw = [5 2 2 3])
end

# ╔═╡ 1db1f070-fb98-4bb6-ab39-4e9e8e1a86cb
md"""
### Speichern des PINN mit meachnistischen Modell
"""

# ╔═╡ ba050844-0b6c-4fc5-b3c3-89548127679a
begin
	pinn_transfer_state = Flux.state( NN_transfer);
	jldsave("pinn_transfer_model.jld2"; pinn_transfer_state)
end

# ╔═╡ d9300313-556c-4219-98ba-09ee0ba4db89
md"""
## Metriken
"""

# ╔═╡ 745e2842-2b51-4dd0-ab08-b4d3a4cd4cb2
r2(y_hat, y) = sum(abs2, y_hat .- mean(y)) / sum(abs2, y .- mean(y))

# ╔═╡ 57d66d4f-29cd-4c22-98ec-99541159cee8
Flux.mse(Tpred_transfer, heating_df.T1), 
Flux.mae(Tpred_transfer, heating_df.T1),
r2(Tpred_transfer, heating_df.T1)

# ╔═╡ e8ddfc47-b313-4a7c-b80d-a78734bac25d
Flux.mse(Tpred_fopdt, heating_df.T1),
Flux.mae(Tpred_fopdt, heating_df.T1),
r2(Tpred_fopdt, heating_df.T1)

# ╔═╡ ca5bb574-a73c-4c62-b9dc-a19f213cbc29
Flux.mse(Tpred_nn, heating_df.T1), 
Flux.mae(Tpred_nn, heating_df.T1),
r2(Tpred_nn, heating_df.T1)

# ╔═╡ 28b8ec62-d9e8-49d3-aab8-efa90210c8bf
md"""
Sowohl mean squared error (mse), als auch mean absolut error (mae) zeigen die selbe Tendenz, dass die Vorhersage des PINN mit FOPDT eine höhere Abweichung von gemessenen Temperaturverlauf hat als die Vorsage des MLP.
Bei jedem Model ist die Korrelation mit dem gemessen Temperaturverlauf gering.
Das höchste r² liegt bei 0.244. Die bessten Ergebnisse werden vom MLP und den PINN-mit Wärmefluss-Modell erreicht.

Die Vorhersage des nODE mit Lux ist leicht besser als des PINN mit FOPDT.
Während die Vorhersage des nODE mit Flux bei MSE, MAE und Bestimmtheitsmaß die größte Abweichung vom tatsächlichen Modell zeigt.

|Netz - physkalischer Loss |  MSE   | MAE   |  r²  |
|:-------------------------| ------ | ----- | ---- |
| MLP                      | 262.48 | 13.18 | 0.208|
|PINN - FODPT              | 311.78 | 15.34 | 0.085|
|PINN - Wärmefluss         | 255.47 | 12.73 | 0.244|
|nODE - Flux               | 313.23 | 15.54 | 0.053|
|nODE - Lux                | 322.57 |15.25  | 0.105|
"""

# ╔═╡ 70b759dc-b50a-41e9-a596-c71460781b6e
md"""

## Änderungen und Fazit

Folgende Anpassungen sind seit der dritten Forlesung erfolgt:

- Anpassung der Eingabe Matrix
- Anpassung des FOPDT-Models
- Einführung des Models mit Wärmefluss
- Einbau von Mini-Batches
- Anpassungen der Beispieldatensatzes für den physikalischen Loss
- Größere Modellen 

Bisher wurden als Eingabematrix eine der vorherige Temperaturwert und der aktuelle Prozentwert der Heizung übergeben. Dies wurde ersetzt durch die Zeit seit dem Start der Messung. Dies erschwerte die Anpassung.

Die Berechnung des Loss beim FOPDT-Modell wurde so angepasst, dass kein Lookup mehr notwenig ist, was die Rechenzeit verkürzt.

Da das bisherige Ergebnis nicht zufriedenstellend war, wurde neben dem FOPDT-Modell noch das Wärmefluss-Modell hinzugefügt ohne deutliche Verbesserungen, gegenüber den klassischen MLP-Modells. Erst der Wechsel auf Minibatches führte zur erfolgreicher Optimierung mit der neuen Eingabematrix bei allen Modellen. Der aufgezeichnete Loss wird glatter, die Berechnung wird beschleunigt.

Beides rührt daher, das nur noch eine Backpropagation pro Batch statt eine Backpropagation pro Messpunkt durchgeführt wird. Damit sinkt der zeitliche Rechenbedarf für die Backpropagation und gleichzeitig, wird die Backpropagation gemittelt über alle Messpunkte des Batches. Die Richtungen der Optimierungsschritte streuen weniger. Je nach Implementierung können auch beim Minibatching weitere Rechenkerne im Feedforward eingesetzt werden, da diese Schritte innerhalb eines Batches unabhängig von einander sind.

Trotz aller Optimimerungen kann eine Verbessungerung des Lernens durch die Verwendung des PINNS nicht immer garantiert werden. Es zeigt sich für die verwendete Methodik, das der physikaische Loss bei FODPT viel starrer während der fortschreiteten Epochen bleibt. Zwar bleibt der pyhsikalische Loss beim Wärmeflussmodell kaum verändert, dennoch lernt das Model oft geringfügig besser als das reine MLP.

Der physikalische Loss wurde erst an Punkten bestimmt, wo die erste, zweite und dritte Ableitung im rollenden Mittel null wurde. Nach der Vorlesung wurden insbesondere zufällig Punkte proportional zum Absolutwert des rollenden Mittelwert der ersten Ableitung der Temperatur bestimmt, da die erste Ableitung die Sensitivtät des Signals vom Eingang beschreibt. Eine direkte Verbessung konnte nicht bestimmt werden. Der physikalische Loss wurde allerding nur für den ersten und fünften Lauf bestimmt. Danach wurde der Loss für den ersten bis einschließlich den fünften Lauf bestimmt. Dies hat keinen eindeutigen positiven Einfluss auf das Training. 

Die erste Optimierung des Netzes ist die Findung der Mittelwert der Temperatur. Daher wurde als zweites Kriterium die Abweichung von dieser Temperatur zu gewählt, um so dass Modell zu unterstützen von der Mittelwerttemperatur abzuweichen. Dies hatte keinen Erfolg.

Bei größeren Modellen wie zum Beispiel fünf Hiddenlayer mit je 32 Knoten. Lernt keines der Modelle mehr als die mittelere Temperatur.
"""

# ╔═╡ 796c53c1-ea69-4e12-bedf-0bf44f600d35
md"""
## Fazit für alle Modelle


Eine Verallgemeinung auf weitere noch nicht trainierte Heizzyklen gelingt mit keinem der Modelle! 
Die benötigten Rechenzeit sind beim PINN und nODE meist viel großer als beim reinen MLP. Dennoch ist die Vorhersage des MLP eine der Besten. Daher lohnt sich für das gegebene Problem der Heizkurve die Verwendung von PINN und nODE nicht.

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RollingFunctions = "b0e4dd01-7b14-53d8-9b45-175a3e362653"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.11"
DataFrames = "~1.5.0"
Flux = "~0.13.16"
JLD2 = "~0.4.31"
NLsolve = "~4.5.1"
PlutoUI = "~0.7.51"
RollingFunctions = "~0.7.0"
StatsBase = "~0.34.0"
StatsPlots = "~0.15.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "e5683b39ba467ddd81f0dd5f48cbdd409056e2b0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "954634616d5846d8e216df1298be2298d55280b2"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.32"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.AccurateArithmetic]]
deps = ["LinearAlgebra", "Random", "VectorizationBase"]
git-tree-sha1 = "07af26e8d08c211ef85918f3e25d4c0990d20d70"
uuid = "22286c92-06ac-501d-9306-4abd417d9753"
version = "0.3.8"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d3f758863a47ceef2248d136657cb9c033603641"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.8"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "89e0654ed8c7aebad6d5ad235d6242c2d737a928"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.3"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "442d989978ed3ff4e174c928ee879dc09d1ef693"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "4.3.2"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "498f45593f6ddc0adff64a9310bb6710e851781b"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.5.0+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "5248d9c45712e51e27ba9b30eebec65658c6ce29"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.6.0+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2918fbffb50e3b7a0b9127617587afa76d4276e8"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "8.8.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "61549d9b52c88df34d21bd306dba1d43bb039c87"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.51.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "a6e6ce44a1e0a781772fc795fb7343b1925e9898"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c72970914c8a21b36bbc244e9df0ed1834a0360b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.95"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "06bf20fcecd258eccf9a6ef7b99856a4dfe7b64c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "589d3d3bff204bdd80ecc53293896b4f39175723"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.1.1"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote", "cuDNN"]
git-tree-sha1 = "64005071944bae14fc145661f617eb68b339189c"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.16"

    [deps.Flux.extensions]
    AMDGPUExt = "AMDGPU"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "478f8c3145bb91d82c2cf20433e8c1b30df454cc"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "a3351bc577a6b49297248aadc23a4add1097c2ac"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.7.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "cb090aea21c6ca78d59672a7e7d13bd56d09de64"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.20.3"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8b8a2fd4536ece6e554168c21860b6820a8a83db"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "19fad9cd9ae44847fe842558a744748084a722d1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.7+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e77dbf117412d4f164a464d610ee6050cc75272"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.6"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "734fd90dd2f920a2f1921d5388dcebe805b262dc"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.14"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "0ec02c648befc2f94156eaef13b0f38106212f3f"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.17"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "42c17b18ced77ff0be65957a591d34f4ed57c631"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.31"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KahanSummation]]
git-tree-sha1 = "6292e7878fe190651e74148edb11356dbbc2e194"
uuid = "8e2b3108-d4c1-50be-a7a2-16352aec75c3"
version = "0.3.1"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "47be64f040a7ece575c2b5f53ca6da7b548d69f4"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "5007c1421563108110bbd57f63d8ad4565808818"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "5.2.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "1222116d7313cdefecf3d45a2bc1a89c4e7c9217"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.22+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "3bb62b5003bc7d2d49f26663484267dc49fa1bf5"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.159"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "FoldsThreads", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "ca31739905ddb08c59758726e22b9e25d0d1521b"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "99e6dbb50d8a96702dc60954569e9fe7291cc55d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.20"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics", "cuDNN"]
git-tree-sha1 = "f94a9684394ff0d325cc12b06da7032d8be01aaf"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "f511fca956ed9e70b80cd3417bb8c2dde4b68644"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.3"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6a01f65dd8583dee82eecc2a19b0ff21521aa749"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.18"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5a6ab2f64388fd1175effdf73fe5933ef1e0bac0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "75ca67b2c6512ad2d0c767a7cfc55e75075f8bbc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.16"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RollingFunctions]]
deps = ["AccurateArithmetic", "KahanSummation", "LinearAlgebra", "LoopVectorization", "Statistics", "StatsBase"]
git-tree-sha1 = "32e8f64eab12eccd7ef87916c1a050291f122ac4"
uuid = "b0e4dd01-7b14-53d8-9b45-175a3e362653"
version = "0.7.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "dbde6766fc677423598138a5951269432b0fcc90"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.7"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "33040351d2403b84afce74dae2e22d3f5b18edcb"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "832afbae2a45b4ae7e831f86965469a24d1d8a83"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.26"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "14ef622cf28b05e38f8af1de57bc9142b03fbfe3"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.5"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "c97f60dd4f2331e1a495527f80d242501d2f9865"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.1"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "25358a5f2384c490e98abd565ed321ffae2cbb37"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.76"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "ba4aa36b2d5c98d6ed1f149da916b3ba46527b2b"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.14.0"
weakdeps = ["InverseFunctions"]

    [deps.Unitful.extensions]
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "ea37e6066bf194ab78f4e747f5245261f17a7175"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.2"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5be3ddb88fc992a7d8ea96c3f10a49a7e98ebc7b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.62"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDNN_jll"]
git-tree-sha1 = "f65490d187861d6222cb38bcbbff3fd949a7ec3e"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.0.4"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─d8760ed7-947f-4a3c-be31-08f67020e769
# ╠═32ac2420-0ad8-11ee-2d6f-2d8c15a32168
# ╟─8733e584-78c6-41c4-9f30-e86ada17fe8f
# ╟─993eda49-5f21-4368-94a6-4801f949a97b
# ╠═ed1920f6-e175-4032-9f8b-eb26a49ea193
# ╠═bf2bea30-98a5-47f4-aa2b-108e88eac174
# ╠═12487f1f-4209-443f-945a-8c85a0191aac
# ╠═95e49920-3c3b-4467-914a-9a93714cc9de
# ╟─0dd3c3d4-3758-4430-a403-96b49a42d44c
# ╠═6a9e72c0-60de-44b4-ab2a-dcf4c6bd2138
# ╟─1a2337f0-bede-4d7d-8fdb-125406dbb703
# ╠═90fa8c5e-e048-4133-b8d3-89116bab7ae4
# ╠═cf7f7ede-39d8-4ab3-b604-6b076e14c978
# ╟─9d979f43-e5bb-47c9-90d2-e6194483e8f3
# ╠═c4d1c1d8-8fa8-441c-9713-9560c12ad2e9
# ╟─757e10e7-1cd2-476c-a189-923aad028ae9
# ╠═c1277d84-5430-4682-a751-b009f4df87d0
# ╠═2824ad6b-b032-42c8-927c-ce6308c7d7bc
# ╟─0f22424a-6147-4326-bed4-65061502d496
# ╠═35aed450-88a5-460a-92b0-39375a28a165
# ╟─2d6f1e50-4bfa-47fa-8ae2-728506db9b19
# ╠═fdf54f26-8169-405e-a288-8247ed275c99
# ╟─a372911c-2c18-495e-9098-4bc76a6700ed
# ╠═9d9ef719-b19b-471e-b6ba-3bcf07fded6f
# ╠═2683688e-e96d-42b3-80cb-ae12ec4d767d
# ╠═d8e34379-6d4b-4149-a2f4-a4944a28f2c4
# ╠═bfb4ace1-1c7f-4305-b83a-f6c7a6feba3f
# ╟─d4b240f7-cbf9-4fa2-9046-bf38661b1d9c
# ╠═5e1c95cc-46e3-4161-aaa1-0232ee06b119
# ╟─cc3cd8e5-5749-4165-b9ee-e3199a0151b3
# ╠═46ae8dc5-e139-4018-97ef-03581ccf6e26
# ╟─99671fef-cd7b-4a60-80f8-8c28ffa173f2
# ╠═96429da4-48a2-4b1d-97d1-7dd3ab6d361c
# ╟─f5c5badd-73a4-465d-90dd-a5caa28e8c92
# ╟─539dc603-636c-49d9-a402-c2e731d9baee
# ╠═552252d9-afb3-407f-a168-d2e7ca62f679
# ╠═73f00d67-63a6-4429-b613-dc3935861459
# ╠═011a7d05-935c-4e1c-a200-391d4108c72a
# ╟─ff6a3eb0-ced6-4901-ab72-5e6a95667e5d
# ╠═2bd87b9f-46dc-492f-9074-a6fd11956270
# ╟─31918ef3-73f2-4f69-9c71-639ba0ecc191
# ╠═9875d167-60c7-4899-a823-41439ffd4fd1
# ╟─7563d048-147a-4fbb-8db2-cfee296a31d3
# ╠═351b6a22-8411-4126-9057-2784c87e3d4f
# ╠═9f52a9c0-6fd4-4d05-97fd-dc80468c7e4f
# ╠═27e6c2de-a08e-4d24-a8b2-62a9c5c4324b
# ╠═015c0f04-85fb-4c2a-82ce-db0ddc848f17
# ╠═4d7ac675-5bad-464b-affa-b3096e8342d8
# ╟─1db1f070-fb98-4bb6-ab39-4e9e8e1a86cb
# ╠═ba050844-0b6c-4fc5-b3c3-89548127679a
# ╠═d9300313-556c-4219-98ba-09ee0ba4db89
# ╠═745e2842-2b51-4dd0-ab08-b4d3a4cd4cb2
# ╠═57d66d4f-29cd-4c22-98ec-99541159cee8
# ╠═e8ddfc47-b313-4a7c-b80d-a78734bac25d
# ╠═ca5bb574-a73c-4c62-b9dc-a19f213cbc29
# ╟─28b8ec62-d9e8-49d3-aab8-efa90210c8bf
# ╟─70b759dc-b50a-41e9-a596-c71460781b6e
# ╟─796c53c1-ea69-4e12-bedf-0bf44f600d35
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
