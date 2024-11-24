extends Node


var xx = Tensor.new([-40., -20.,  0., 20.,  40.,  60.,  80.])#Celsius
var yy = Tensor.new([-40.,  -4., 32., 68., 104., 140., 176.])#Fahrenheit

var seed = hash("G-Mind")

var Net = Sequential.new([
	Dense.create(1, 1, self.seed),
	])


var mse_loss = MSELoss.new()

var c_scaler = MinMaxScaler.new()
var f_scaler = MinMaxScaler.new()


# Called when the node enters the scene tree for the first time.
func _ready():
	self.c_scaler.fit(xx)
	self.f_scaler.fit(yy)
	xx = self.c_scaler.transform(xx)
	yy = self.f_scaler.transform(yy)

	print("---------------------------------")
	print(xx)
	print(yy)
	print("---------------------------------")

	# xx = self.c_scaler.inverse_transform(xx)
	# yy = self.f_scaler.inverse_transform(yy)


func train():
	var epochs = 1000.
	var batch_size = 1
	var learning_rate = 0.01

	var count = 0

	for epoch in range(epochs):
		var epoch_loss = 0.

		for i in range(len(xx.values)):
			var x = Tensor.new([ xx.values[i] ])
			var y = Tensor.new([ yy.values[i] ])

			var y_hat = Net.forward(x)

			var loss: Tensor = mse_loss.forward(y_hat, y)
			epoch_loss += loss.values[0]
			loss.backward(batch_size)

			count += 1

			Net.clip_gradients(1.0)

			if count % batch_size == 0:
				Net.step(learning_rate, true)


		if (epoch + 1) % 10 == 0:
			print("Epoch [", epoch + 1., "] Loss: ", epoch_loss / len(xx.values))
	

func do_pred():
	var x = Tensor.new([75./100.])#Celsius

	var pred: Tensor = Net.forward(x)

	print("X = {x_values} -- Y = 167 -- Y_HAT = {y_hat_values}".format(
		{ "x_values": x.values[0] * 100., "y_hat_values": pred.values[0] * 100. }
		))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_celsius_input_text_changed():
	var celsius = $calculator_container/celsius_container/celsius_input.text
	
	var x: Tensor = c_scaler.transform(Tensor.new([float(celsius)]))

	var pred: Tensor = Net.forward(x)
	var y = f_scaler.inverse_transform(pred)
	
	$calculator_container/fahrenheit_container/n_fahrenheit.text = str(y.values[0])


func _on_train_pressed():
	self.train()
	$model_container/train.toggle_mode = false
