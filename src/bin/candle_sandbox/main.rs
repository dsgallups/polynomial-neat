use candle_core::{Device, Result, Tensor};

struct Model {
    first: Tensor,
    second: Tensor,
}

impl Model {
    fn example(device: &Device) -> Result<Self> {
        let first = Tensor::randn(0f32, 1.0, (784, 100), device)?;
        let second = Tensor::randn(0f32, 1.0, (100, 10), device)?;
        Ok(Self { first, second })
    }
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = image.matmul(&self.first)?;
        let x = x.relu()?;
        x.matmul(&self.second)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model = Model::example(&device)?;

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit = model.forward(&dummy_image)?;

    println!("digit: {:?}", digit);

    Ok(())
}
