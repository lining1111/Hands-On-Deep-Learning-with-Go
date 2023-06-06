package main

import (
	. "gorgonia.org/gorgonia"
)

type LSTM struct {
	//input
	wix    *Node
	wih    *Node
	bias_i *Node
	//forget gate weights
	wfx    *Node
	wfh    *Node
	bias_f *Node
	//output gate weights
	wox    *Node
	woh    *Node
	bias_o *Node
	//cell write
	wcx    *Node
	wch    *Node
	bias_c *Node
}

func MakeLSTM(g *ExprGraph, hiddenSize, prevSize int) LSTM {
	retVal := LSTM{}

	retVal.wix = NewMatrix(g, Float, WithShape(hiddenSize, prevSize), WithInit(GlorotN(1.0)), WithName("wix_"))
	retVal.wih = NewMatrix(g, Float, WithShape(hiddenSize, hiddenSize), WithInit(GlorotN(1.0)), WithName("wih_"))
	retVal.bias_i = NewVector(g, Float, WithShape(hiddenSize), WithName("bias_i_"), WithInit(Zeroes()))

	// forget gate weights
	retVal.wox = NewMatrix(g, Float, WithShape(hiddenSize, prevSize), WithInit(GlorotN(1.0)), WithName("wfx_"))
	retVal.woh = NewMatrix(g, Float, WithShape(hiddenSize, hiddenSize), WithInit(GlorotN(1.0)), WithName("wfh_"))
	retVal.bias_o = NewVector(g, Float, WithShape(hiddenSize), WithName("bias_f_"), WithInit(Zeroes()))

	// output gate weights

	retVal.wfx = NewMatrix(g, Float, WithShape(hiddenSize, prevSize), WithInit(GlorotN(1.0)), WithName("wox_"))
	retVal.wfh = NewMatrix(g, Float, WithShape(hiddenSize, hiddenSize), WithInit(GlorotN(1.0)), WithName("woh_"))
	retVal.bias_f = NewVector(g, Float, WithShape(hiddenSize), WithName("bias_o_"), WithInit(Zeroes()))

	// cell write

	retVal.wcx = NewMatrix(g, Float, WithShape(hiddenSize, prevSize), WithInit(GlorotN(1.0)), WithName("wcx_"))
	retVal.wch = NewMatrix(g, Float, WithShape(hiddenSize, hiddenSize), WithInit(GlorotN(1.0)), WithName("wch_"))
	retVal.bias_c = NewVector(g, Float, WithShape(hiddenSize), WithName("bias_c_"), WithInit(Zeroes()))
	return retVal
}

func (l *LSTM) learnables() Nodes {
	return Nodes{
		l.wix, l.wih, l.bias_i,
		l.wfx, l.wfh, l.bias_f,
		l.wcx, l.wch, l.bias_c,
		l.wox, l.woh, l.bias_o,
	}
}

//Activate is used to define the operations our units perform when processing input data
//LSTM的机制 inputVector 输入节点 ---prev 上次lstm的输出，包含 hidden和cell
//h0 h1 inputGate 第一层运算 1
//					h0 = LSTM.wix输入层x	*	inputVector	;
//					h1 = LSTM.wih输入层h	*	prevHidden ;
//					inputGate = Sigmoid(h0+h1+LSTM.bias_i)
//h2 h3 forgetGate 第一层运算 2
//					h2 = LSTM.wfx遗忘层x	*	inputVector	;
//					h3 = LSTM.wfh遗忘层h	*	prevHidden ;
//					forgetGate = Sigmoid(h2+h3+LSTM.bias_f)
//h4 h5 outputGate 第一层运算 3
//					h2 = LSTM.wox输出层x	*	inputVector	;
//					h3 = LSTM.woh输出层h	*	prevHidden ;
//					outputGate = Sigmoid(h4+h5+LSTM.bias_o)
//h6 h7 cellWrite 第一层运算 4
//					h6 = LSTM.wcx写入层x	*	inputVector	;
//					h7 = LSTM.wch写入层h	*	prevHidden ;
//					cellWrite = Tanh(h4+h5+LSTM.bias_c)
//retain, write 第二层运算 1
//retain = HadamardProd(forgetGate,prev.cell) //遗忘层输出与上一层的cell做哈达玛运算(相同位置相乘)
//write = HadamardProd(inputGate, cellWrite) //输入层输出与这层的cellWrite做哈达玛运算(相同位置相乘)
// 本层的cell = retain+write //加法
//本层的hidden = HadamardProd(outputGate, Tanh(cell))// 相同位置乘
func (l *LSTM) Activate(inputVector *Node, prev lstmout) (out lstmout, err error) {
	// log.Printf("prev %v", prev.hidden.Shape())
	prevHidden := prev.hidden
	prevCell := prev.cell

	var h0, h1, inputGate *Node
	h0 = Must(Mul(l.wix, inputVector))
	h1 = Must(Mul(l.wih, prevHidden))
	inputGate = Must(Sigmoid(Must(Add(Must(Add(h0, h1)), l.bias_i))))

	var h2, h3, forgetGate *Node
	h2 = Must(Mul(l.wfx, inputVector))
	h3 = Must(Mul(l.wfh, prevHidden))
	forgetGate = Must(Sigmoid(Must(Add(Must(Add(h2, h3)), l.bias_f))))

	var h4, h5, outputGate *Node
	h4 = Must(Mul(l.wox, inputVector))
	h5 = Must(Mul(l.woh, prevHidden))
	outputGate = Must(Sigmoid(Must(Add(Must(Add(h4, h5)), l.bias_o))))

	var h6, h7, cellWrite *Node
	h6 = Must(Mul(l.wcx, inputVector))
	h7 = Must(Mul(l.wch, prevHidden))
	cellWrite = Must(Tanh(Must(Add(Must(Add(h6, h7)), l.bias_c))))

	// cell activations
	var retain, write *Node
	retain = Must(HadamardProd(forgetGate, prevCell))
	write = Must(HadamardProd(inputGate, cellWrite))
	cell := Must(Add(retain, write))
	hidden := Must(HadamardProd(outputGate, Must(Tanh(cell))))
	out = lstmout{
		hidden: hidden,
		cell:   cell,
	}
	return
}

type lstmout struct {
	hidden, cell *Node
}
