use rand::Rng;
use num_format::{Locale, ToFormattedString};
use std::io::{self, Write};
mod ml;

/// Returns the input (a float) of the user. Will ask until the input is a number.
/// # Arguments
/// * `prompt` - A string describing the data asked.
fn get_input(prompt : &str) -> f64 {
	loop {

		let mut user_input = String::new();
		print!("{prompt}: ");
		io::stdout().flush().unwrap();

		io::stdin()
			.read_line(&mut user_input)
			.expect("Can't read input");

		let user_input:f64 = match user_input.trim().parse() {
			Ok(num)=>num,
			Err(_)=>{
				println!("Please input a number");
				continue;
			}
		};
		return user_input;
	}
}

fn main() {
	let mut train_batch:Vec<(f64,f64)> = Vec::new();
	let mut test_batch:Vec<(f64,f64)> = Vec::new();

	// the data set parameters
	let data_set_w;
	let data_set_b;
	let noise_range;
	let data_set_input_min = -255.0;
	let data_set_input_max = 255.0;
	// the size of the generated data set
	let mut data_len;


	/********* Asking the user for parameters *********/

	println!("Please enter the weight and bias parameter to generate the data set");
	data_set_w = get_input("weight parameter");
	data_set_b = get_input("bias parameter");
	println!("Now enter the noise desired for the parameter (as a float, put 0 for disabling noise)");
	noise_range = get_input("noise").abs();
	println!("Last step, can you please provide the data set size");
	loop {
		data_len = get_input("data_set size") as usize;
		if data_len<=0 {
			println!("please provide a non-null positive size!");
			continue;
		}
		if data_len > 1_000_000_000 {
			data_len = 1_000_000_000;
		}
		break;
	}




	// model hyperparameters
	let learning_rate:f64 = 1e-5;
	let data_chunk = 10;

	// randomized weight and bias for training the model
	let untrained_w = rand::thread_rng().gen_range(-100.0..100.0);
	let untrained_b = rand::thread_rng().gen_range(-100.0..100.0);


	// divide the data set in a training set and a testing set (80% for trainging, 20% for testing)
	let training_data_len = data_len * 80/100;
	let testing_data_len = data_len - training_data_len;


	println!("Generating the data set ...");
	for _ in 0..training_data_len {
		// generate an input base on the given range
		let random_input = rand::thread_rng().gen_range(data_set_input_min..=data_set_input_max);

		// we can add a noise by setting epsilon_noise() parameter to anything except 0, ussally this will result in an average error of half of the range in a well trained model
		train_batch.push((random_input, random_input* data_set_w + data_set_b + ml::epsilon_noise(noise_range)));
	}
	for _ in 0..testing_data_len {
		// generate an input base on the given range
		let random_input = rand::thread_rng().gen_range(data_set_input_min..=data_set_input_max);

		// add a noise free output for testing
		test_batch.push((random_input, random_input* data_set_w + data_set_b));
	}

	println!("Data set generated ! \nTraining the model...");



	// train the model
	let (w,b) = ml::train(untrained_w,untrained_b, &train_batch, learning_rate,data_chunk,true);
	// test the model
	let avr_err = ml::test_model(w,b,&test_batch);

	println!("\n[{:=^50}]"," Data Set ");
	let mut display_string = format!("data_set size: {}",data_len.to_formatted_string(&Locale::fr));
	println!("{: ^50}",display_string);
	display_string = format!("training_set size: {}",train_batch.len().to_formatted_string(&Locale::fr));
	println!("{: ^50}",display_string);
	display_string = format!("testing_set size: {}",test_batch.len().to_formatted_string(&Locale::fr));
	println!("{: ^50}",display_string);
	display_string = format!("data_set w: {}", data_set_w);
	println!("{: ^50}",display_string);
	display_string = format!("data_set b: {}", data_set_b);
	println!("{: ^50}",display_string);
	display_string = format!("data_set input range: [{data_set_input_min} ; {data_set_input_max}]");
	println!("{: ^50}",display_string);
	display_string = format!("data_set noise range: [-{noise_range} ; {noise_range}]");
	println!("{: ^50}",display_string);



	println!("\n[{:=^50}]"," Parameters ");
	display_string = format!("w: {:.5}",untrained_w);
	println!("{: ^50}",display_string);
	display_string = format!("b: {:.5}",untrained_b);
	println!("{: ^50}",display_string);
	display_string = format!("learning_rate: {:.5}",learning_rate);
	println!("{: ^50}",display_string);
	display_string = format!("data_chunk: {}",data_chunk);
	println!("{: ^50}",display_string);

	println!("\n[{:=^50}]"," Results ");
	display_string = format!("w [TRAINED]: {:.5} (distance : {:.5})",w, (data_set_w - w).abs());
	println!("{: ^50}",display_string);
	display_string = format!("b [TRAINED]: {:.5} (distance : {:.5})",b, (data_set_b - b).abs());
	println!("{: ^50}",display_string);
	display_string = format!("cost: {}", ml::fn_cost(w,b,&test_batch));
	println!("{: ^50}",display_string);
	display_string = format!("avr_error: {:.8}",avr_err);
	println!("{: ^50}",display_string);



}