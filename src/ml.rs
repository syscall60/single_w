// TEST
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_test() {
		for _ in 0..=1000 {
			let random_range = rand::thread_rng().gen_range(0.0..=1000.0);
			let noise = epsilon_noise(random_range);
			assert!(-random_range<= noise && noise <=random_range);
		}
    }
}


/// Returns a string of a progress bar based on the percentage and the size of the bar in parameters
/// # Arguments
/// * `percentage` - the percentage fo the operation
/// * `size` - the size of the progress bar (characters on the terminal) 
fn progress_bar(percentage: f64, size: usize) -> String {
	// the full part of the progress bar
	let repeat = percentage * size as f64 / 100.0;

	let full = "█".repeat(repeat.round() as usize);
	let empty = "▁".repeat(size - (repeat.round() as usize));

	[full, empty].join("")
}

/// Display the current phase of the training (with a progress bar)
/// # Arguments
/// * `current_step` - the current batch
/// * `last` - the number of batches
/// * `w` - the current weight
/// * `b` - the current bias
/// * `cost` - the current cost
fn display_progress(current_step: i32,last: usize,w: f64,b: f64,cost: f64) -> (){
	let percentage:f64 = current_step as f64 * 100.0 / last as f64;

	println!("{} {:.2}% ",progress_bar(percentage,50),percentage);
	println!("w: {:.8}",w);
	println!("b: {:.8}",b);
	println!("cost: {:.8}",cost)
}

use rand::Rng;

/// Returns the average cost (the error saquared) of a given data set based on parameters `w` (weight) and `b` (bias)
/// 
/// # Arguments
/// * `w` - weight of the model
/// * `b` - bias of the model
/// * `data` - the data set in this form : (`input`,`output`)
pub fn fn_cost(w : f64, b:f64, data : &[(f64,f64)]) -> f64{

	// accumulative cost
	let mut result = 0.0; 

	for (input,output)  in data {
		let computed_value = input*w +b;
		let distance = (computed_value-output).powi(2);
		result += distance;
	}

	result/(data.len() as f64)
}


/// Returns a tuple (**`dw`** : _partial derivative of w_ ,**`db`** _partial derivative of b_) containing the average gradient of the cost function (i.e. the steps to take for `w` (the weight) and `b` (the bias) to decrease the cost) based on `data` the data set chunk given
/// 
/// # Arguments
/// * `w` - the current weight of the model
/// * `b` - the current bias of the model
/// * `data` - the data set in this form : (`input`,`output`)
fn cost_derivative(w : f64,b:f64, data: &[(f64,f64)]) -> (f64,f64){

	//accumulative gradient
	let mut w_result = 0.0;
	let mut b_result = 0.0;

	for (input,output)  in data {
		let w_distance = input*(w*input-output+b); //partial derivative in w of (w * input + b - output)² divided by 2 as it will be multipled by a learning rate anyway
		let b_distance = w*input-output+b; //partial derivative in b of (w * input + b - output)² divided by 2 as it will be multipled by a learning rate anyway
		w_result += w_distance;
		b_result += b_distance;
	}

	(w_result/(data.len() as f64),b_result/(data.len() as f64))
}

/// Returns a tuple (`w`,`b`) of the trained weight and bias on the data set `data`. **Highly sensitive to the `learning_rate`**
/// 
/// # Arguments
/// * `w` - randomized weight of the model
/// * `b` - randomized bias of the model
/// * `data` - the data set in this form : (`input`,`output`)
/// * `learning_data` - the learning rate of the model (i.e. the "length" of each "step" in the gradient descent). **This parameter is crucial**, is the model is not performing well, consider increase of decrease the learning rate (usually between 1 and 1e-6)
/// * `data_chunk` - the data will be sliced in chunks to optimized the memory when traning to average only on the chunk and not the entire data set in one go. This value is usually at 10.
/// * `verbose` - if we display progress or not
pub fn train(mut w:f64,mut b:f64, data: &[(f64,f64)], learning_rate :f64,data_chunk:usize,verbose : bool) -> (f64,f64){
	//slice the data into chunks
	let data_chunks = data.chunks(data_chunk);

	// to keep track of the current step (only for display)
	let mut i = 0;
	let chunks_size = data_chunks.len();


	let mut cost = fn_cost(w,b,data);
	
	if verbose {
		display_progress(i, chunks_size, w, b, cost);
	}

	for data in data_chunks {
		//using the gradient of the cost function to update w and b
		let (w_correction,b_correction) = cost_derivative(w,b,data);
		w = w - w_correction*learning_rate;
		b = b - b_correction*learning_rate;
		
		// the updated cost (only for display)
		cost = fn_cost(w,b,data);
		i+=1;

		if verbose && i%500==0 {
			println!("\x1b[5F");
			display_progress(i, chunks_size, w, b, cost);
		}
	}

	if verbose {
		println!("\x1b[5F");
		display_progress(i, chunks_size, w, b, cost);
	}




	(w,b)
}

/// Returns a random uniform distributed float between -range and range (inclusive). It is used to generate the noise of data set
/// 
/// # Arguments
/// * `range` - the range of generation
/// 
/// # Examples
/// ```
/// 
/// let noise = epsilon_noise(10.0);
/// assert!(-10.0 <= noise && noise <= 10.0);
/// ```
pub fn epsilon_noise(range:f64)->f64{
	rand::thread_rng().gen_range(-range..=range)
}

/// Returns the average error (distance based of the absolute difference between input nad output) of a model based on the data set `data` (usually reserved to test)
/// # Arguments
/// * `w` - the trained weight of the model
/// * `b` - the trained bias of the model
/// * `data` - the data set in this form : (`input`,`output`)
pub fn test_model(w: f64,b:f64,data: &[(f64,f64)]) -> f64
{
	let mut accumlative_error = 0.0;
	for (input,output) in data {
		let prediction = w*input+b;

		accumlative_error += (prediction-output).abs();
	}

	accumlative_error/(data.len() as f64)
}