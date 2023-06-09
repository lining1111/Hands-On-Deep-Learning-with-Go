package main

import (
	"fmt"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var Float = tensor.Float32

type contextualError interface {
	error
	Node() *Node
	Value() Value
	InstructionID() int
	Err() error
}

type GRU struct {

	// weights for mem
	u *Node
	w *Node
	b *Node

	// update gate
	uz *Node
	wz *Node
	bz *Node

	// reset gate
	ur  *Node
	wr  *Node
	br  *Node
	one *Node

	Name string // optional name
}

func MakeGRU(name string, g *ExprGraph, inputSize, hiddenSize int, dt tensor.Dtype) GRU {
	// standard weights
	u := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.u", name)), WithInit(GlorotN(1.0)))
	w := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.w", name)), WithInit(GlorotN(1.0)))
	b := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.b", name)), WithInit(Zeroes()))

	// update gate
	uz := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.uz", name)), WithInit(GlorotN(1.0)))
	wz := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.wz", name)), WithInit(GlorotN(1.0)))
	bz := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.b_uz", name)), WithInit(Zeroes()))

	// reset gate
	ur := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.ur", name)), WithInit(GlorotN(1.0)))
	wr := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.wr", name)), WithInit(GlorotN(1.0)))
	br := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.bz", name)), WithInit(Zeroes()))

	ones := tensor.Ones(dt, hiddenSize)
	one := g.Constant(ones)
	gru := GRU{
		u: u,
		w: w,
		b: b,

		uz: uz,
		wz: wz,
		bz: bz,

		ur: ur,
		wr: wr,
		br: br,

		one: one,
	}
	return gru
}

//Activate
// x：这次参与运算的 prev:上一次节点的输出
//uz wz  第一层运算1
//		uzh = uz	*	prev
//		wzx = wz	*	x
//		z = uzh+wzx + bz(偏置)
//ur wr  第一层运算2
//		urh = ur	*	prev
//		wrx = wr	*	x
//		r = urh+wrx + br(偏置)
//u 第二层运算1
//		hiddenFilter = u	*	（哈达玛积(r,prev)）
//		wx = w	*	x
//		第三层运算得到返回节点
func (l *GRU) Activate(x, prev *Node) (retVal *Node, err error) {
	// update gate
	uzh := Must(Mul(l.uz, prev))
	wzx := Must(Mul(l.wz, x))
	z := Must(Sigmoid(
		Must(Add(
			Must(Add(uzh, wzx)),
			l.bz))))

	// reset gate
	urh := Must(Mul(l.ur, prev))
	wrx := Must(Mul(l.wr, x))
	r := Must(Sigmoid(
		Must(Add(
			Must(Add(urh, wrx)),
			l.br))))

	// memory for hidden
	hiddenFilter := Must(Mul(l.u, Must(HadamardProd(r, prev))))
	wx := Must(Mul(l.w, x))
	mem := Must(Tanh(
		Must(Add(
			Must(Add(hiddenFilter, wx)),
			l.b))))

	omz := Must(Sub(l.one, z))
	omzh := Must(HadamardProd(omz, prev))
	upd := Must(HadamardProd(z, mem))
	retVal = Must(Add(upd, omzh))
	return
}

func (l *GRU) learnables() Nodes {
	retVal := make(Nodes, 0, 9)
	retVal = append(retVal, l.u, l.w, l.b, l.uz, l.wz, l.bz, l.ur, l.wr, l.br)
	return retVal
}
