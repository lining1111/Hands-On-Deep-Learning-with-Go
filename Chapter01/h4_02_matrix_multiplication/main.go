package main

import (
	"fmt"
	"io/ioutil"
	"log"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

//矩阵运算
func main() {
	g := G.NewGraph()

	matB := []float32{0.9, 0.7, 0.4, 0.2}
	matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
	mat := G.NewMatrix(g,
		tensor.Float32,
		G.WithName("W"),
		G.WithShape(2, 2),
		G.WithValue(matT),
	)

	vecB := []float32{5, 7}
	vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(2))

	vec := G.NewVector(g,
		tensor.Float32,
		G.WithName("x"),
		G.WithShape(2),
		G.WithValue(vecT),
	)

	z, err := G.Mul(mat, vec) //z=W*x

	// create a VM to run the program on
	machine := G.NewTapeMachine(g)

	// set initial values then run

	if machine.RunAll() != nil {
		log.Fatal(err)
	}

	fmt.Println(z.Value().Data())
	// Output: [9.4 3.4]
	//让我们通过图来看看g是什么样子 .dot 文件通过 dot -T pdf multiplication.dot -o multiplication.pdf 命令转换
	ioutil.WriteFile("multiplication.dot", []byte(g.ToDot()), 0644)
}
