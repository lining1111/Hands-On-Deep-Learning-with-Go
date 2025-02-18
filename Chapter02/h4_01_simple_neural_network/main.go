package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var err error

type nn struct {
	g      *ExprGraph
	w0, w1 *Node

	pred    *Node
	predVal Value
}

func newNN(g *ExprGraph) *nn {
	// Create node for w/weight
	//wB :=[]float64{-0.168,0.441,-1}
	wB := tensor.Random(tensor.Float64, 3)
	wT := tensor.New(tensor.WithBacking(wB), tensor.WithShape(3, 1))
	w0 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(3, 1),
		WithValue(wT),
	)
	return &nn{
		g:  g,
		w0: w0,
	}
}

func (m *nn) learnables() Nodes {
	return Nodes{m.w0}
}

//隐藏层操作
func (m *nn) fwd(x *Node) (err error) {
	var l0, l1 *Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for Sigmoid
	l0dot := Must(Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = Must(Sigmoid(l0dot))

	m.pred = l1
	Read(m.pred, &m.predVal)
	return nil

}

//表征化上面的Sigmoid函数
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(x))
}

//本例中输入是4*3 输出 4*1 ，神经元内有两层 第一层是3*1,第二层是一个激活函数，其实输出的元素值是0和1,像一个分类器
//输入	0 0 1
//		0 1 1
//		1 0 1
//		1 1 1
//输出	0
//		0
//		1
//		1
func main() {

	rand.Seed(31337)

	// Create graph and network
	g := NewGraph()
	m := newNN(g)

	// Set input x to network
	xB := []float64{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1}
	xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(4, 3))
	x := NewMatrix(g,
		tensor.Float64,
		WithName("X"),
		WithShape(4, 3),
		WithValue(xT),
	)

	// Define validation data set
	yB := []float64{0, 0, 1, 1}
	yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(4, 1))
	y := NewMatrix(g,
		tensor.Float64,
		WithName("y"),
		WithShape(4, 1),
		WithValue(yT),
	)

	// Run forward pass 计算出原始数据通过神经元后的输出、用于和实际的输出做对比
	if err := m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// Calculate Cost w/MSE
	losses := Must(Sub(y, m.pred)) //差距

	cost1 := Must(Mean(losses))
	var costVal Value
	Read(cost1, &costVal)

	ioutil.WriteFile("pregrad.dot", []byte(g.ToDot()), 0644)

	square := Must(Square(losses))
	cost := Must(Mean(square)) //差距的方差

	// Do Gradient updates计算梯度
	if _, err = Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// Instantiate VM and Solver朴素随机梯度下降
	vm := NewTapeMachine(g, BindDualValues(m.learnables()...))
	//学习率：太小导致训练的周期长、太大则结果很快就越过去了
	solver := NewVanillaSolver(WithLearnRate(0.1), WithClip(5)) //W_New=W_old -eta*derivative(导数)
	//solver1 := NewMomentum()
	for i := 0; i < 10000; i++ {
		vm.Reset()
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", i, err)
		}
		solver.Step(NodesToValueGrads(m.learnables()))
		fmt.Println("\nState at iter ", i)
		fmt.Println("Cost: \n", cost.Value())
		fmt.Println("Weights: \n", m.w0.Value())
		//vm.Reset()
	}
	fmt.Println("\n\nOutput after Training: \n", m.predVal)
}
