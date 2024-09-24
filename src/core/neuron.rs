use crate::prelude::*;

pub struct NeuronInner<N, I> {
    pub(crate) inner: N,
    props: Option<Props<I>>,
}

impl<N, I> NeuronInner<N, I> {
    pub fn new(inner: N, props: Option<Props<I>>) -> Self {
        Self { inner, props }
    }

    pub fn inputs(&self) -> Option<&[Input<I>]> {
        self.props.as_ref().map(|props| props.inputs())
    }

    pub fn props(&self) -> Option<&Props<I>> {
        self.props.as_ref()
    }

    pub fn neuron_type(&self) -> NeuronType {
        match self.props {
            None => NeuronType::input(),
            Some(ref props) => props.props_type().into(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &N {
        &self.inner
    }

    pub fn is_input(&self) -> bool {
        self.neuron_type() == NeuronType::input()
    }
    pub fn is_hidden(&self) -> bool {
        self.neuron_type() == NeuronType::hidden()
    }
    pub fn is_output(&self) -> bool {
        self.neuron_type() == NeuronType::output()
    }
}

pub trait Neuron<'a, N, I>
where
    N: 'a,
    I: 'a,
{
    fn inner(&self) -> &NeuronInner<N, I>;

    fn inputs(&'a self) -> Option<&[Input<I>]> {
        self.inner().inputs()
    }

    fn props(&'a self) -> Option<&Props<I>> {
        self.inner().props()
    }

    fn neuron_type(&'a self) -> NeuronType {
        self.inner().neuron_type()
    }

    fn is_input(&'a self) -> bool {
        self.inner().is_input()
    }
    fn is_hidden(&'a self) -> bool {
        self.inner().is_hidden()
    }
    fn is_output(&'a self) -> bool {
        self.inner().is_output()
    }
}
