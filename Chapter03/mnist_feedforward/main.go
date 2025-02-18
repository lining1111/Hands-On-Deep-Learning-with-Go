package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"gopkg.in/cheggaaa/pb.v1"
	"image"
	"image/jpeg"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	_ "net/http/pprof"

	"github.com/m8u/gorgonia/examples/mnist"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"time"
)

var (
	epochs     = flag.Int("epochs", 100, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 100, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "../../dataset/mnist/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type nn struct {
	g          *gorgonia.ExprGraph
	w0, w1, w2 *gorgonia.Node

	out     *gorgonia.Node
	predVal gorgonia.Value
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func newNN(g *gorgonia.ExprGraph) *nn {
	// Create node for w/weight
	w0 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(784, 300), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(300, 100), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(100, 10), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
	}
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2}
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var l0, l1, l2 *gorgonia.Node
	var l0dot, l1dot *gorgonia.Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for ReLU
	if l0dot, err = gorgonia.Mul(l0, m.w0); err != nil {
		return errors.Wrap(err, "Unable to multiply l0 and w0")
	}

	// l0dot := gorgonia.Must(gorgonia.Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = gorgonia.Must(gorgonia.Rectify(l0dot))

	// MOAR layers
	// l2dot := gorgonia.Must(gorgonia.Mul(l1, m.w1))

	if l1dot, err = gorgonia.Mul(l1, m.w1); err != nil {
		return errors.Wrap(err, "Unable to multiply l1 and w1")
	}
	l2 = gorgonia.Must(gorgonia.Rectify(l1dot))

	var out *gorgonia.Node
	if out, err = gorgonia.Mul(l2, m.w2); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w2")
	}

	// m.pred = l3dot
	// gorgonia.Read(m.pred, &m.predVal)
	// return nil

	m.out, err = gorgonia.SoftMax(out)
	gorgonia.Read(m.out, &m.predVal)
	return

}

const pixelRange = 255

func reversePixelWeight(px float64) byte {
	// return byte((pixelRange*px - pixelRange) / 0.9)
	return byte(pixelRange*math.Min(0.99, math.Max(0.01, px)) - pixelRange)
}

func visualizeRow(x []float64) *image.Gray {
	// since this is a square, we can take advantage of that
	l := len(x)
	side := int(math.Sqrt(float64(l)))
	r := image.Rect(0, 0, side, side)
	img := image.NewGray(r)

	pix := make([]byte, l)
	for i, px := range x {
		pix[i] = reversePixelWeight(px)
	}
	img.Pix = pix

	return img
}

//输入量为28*28矩阵，或者784(784=28*28)的向量,即图片的像素值
//输出为10*1的向量。每个输入对应的输出向量上只有一个为1,来对应数字的0～9
//定义神经网络有两个隐藏层，一个有300个单元，一个有100个单元。
func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(7945)

	// // intercept Ctrl+C
	// sigChan := make(chan os.Signal, 1)
	// signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// doneChan := make(chan bool, 1)

	var inputs, targets tensor.Tensor
	var err error

	// load our data set
	trainOn := *dataset
	if inputs, targets, err = mnist.Load(trainOn, loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples := inputs.Shape()[0]
	bs := *batchsize

	// MNIST data consists of 28 by 28 black and white images
	// however we've imported it directly now as 784 different pixels
	// as a result, we need to reshape it to match what we actually want
	// if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
	// 	log.Fatal(err)
	// }

	//以Graph为整体，通过NewXX加入Node来描述节点的特性
	// we should now also proceed to put in our desired variables
	// x is where our input should go, while y is the desired output
	g := gorgonia.NewGraph()
	// x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	//这里通过bs 将输入x 和输出y都做了扩容，但是属于行上的扩充，为了加快速度，每行的输入，对应每行的输出。
	x := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"))

	// ioutil.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)

	m := newNN(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// ioutil.WriteFile("simple_graph_2.dot", []byte(g.ToDot()), 0644)
	//loss = -1*mean(actual_y*predicted_y)
	//对应位置相乘 哈达玛积。
	losses, err := gorgonia.HadamardProd(m.out, y)
	if err != nil {
		log.Fatal(err)
	}
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	// we wanna track costs
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	//前台打印的状态
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		// for i := 0; i < 1; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}
			// if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
			// 	log.Fatal("Unable to reshape %v", err)
			// }
			if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
				log.Fatal("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}
			// solver.Step(m.learnables())
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		bar.Update()
		log.Printf("Epoch %d | cost %v", i, costVal)
	}
	bar.Finish()

	log.Printf("Run Tests")

	// load our test set
	if inputs, targets, err = mnist.Load("test", loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples = inputs.Shape()[0]
	bs = *batchsize
	batches = numExamples / bs

	bar = pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)
	bar.Prefix(fmt.Sprintf("Epoch Test"))
	bar.Set(0)
	bar.Start()
	for b := 0; b < batches; b++ {
		start := b * bs
		end := start + bs
		if start >= numExamples {
			break
		}
		if end > numExamples {
			end = numExamples
		}

		var xVal, yVal tensor.Tensor
		if xVal, err = inputs.Slice(sli{start, end}); err != nil {
			log.Fatal("Unable to slice x")
		}

		if yVal, err = targets.Slice(sli{start, end}); err != nil {
			log.Fatal("Unable to slice y")
		}
		// if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
		// 	log.Fatal("Unable to reshape %v", err)
		// }
		if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
			log.Fatal("Unable to reshape %v", err)
		}

		gorgonia.Let(x, xVal)
		gorgonia.Let(y, yVal)
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at epoch test: %v", err)
		}

		arrayOutput := m.predVal.Data().([]float64)
		yOutput := tensor.New(tensor.WithShape(bs, 10), tensor.WithBacking(arrayOutput))

		for j := 0; j < xVal.Shape()[0]; j++ {
			rowT, _ := xVal.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			// get label
			yRowT, _ := yVal.Slice(sli{j, j + 1})
			yRow := yRowT.Data().([]float64)
			var rowLabel int
			var yRowHigh float64

			for k := 0; k < 10; k++ {
				if k == 0 {
					rowLabel = 0
					yRowHigh = yRow[k]
				} else if yRow[k] > yRowHigh {
					rowLabel = k
					yRowHigh = yRow[k]
				}
			}

			// get prediction
			predRowT, _ := yOutput.Slice(sli{j, j + 1})
			predRow := predRowT.Data().([]float64) //模型预测的结果1*10的向量
			var rowGuess int
			var predRowHigh float64

			// guess result 遍历当前的结果向量，得到得分值最大的那个
			for k := 0; k < 10; k++ {
				if k == 0 {
					rowGuess = 0
					predRowHigh = predRow[k]
				} else if predRow[k] > predRowHigh {
					rowGuess = k
					predRowHigh = predRow[k]
				}
			}
			//b 批次号码 j条目号码 rowLabel MNIST提供的label rowGuess 模型的预测
			f, _ := os.OpenFile(fmt.Sprintf("images/%d-%d-%d-%d.jpg", b, j, rowLabel, rowGuess), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
			err = f.Close()
			if err != nil {
				log.Println(err)
			}
		}

		arrayOutput = m.predVal.Data().([]float64)
		yOutput = tensor.New(tensor.WithShape(bs, 10), tensor.WithBacking(arrayOutput))

		file, err := os.OpenFile(fmt.Sprintf("csv/%d.csv", b), os.O_CREATE|os.O_WRONLY, 0777)
		if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
			log.Fatal("Unable to create csv", err)
		}
		defer file.Close()
		var matrixToWrite [][]string

		for j := 0; j < yOutput.Shape()[0]; j++ {
			rowT, _ := yOutput.Slice(sli{j, j + 1}) //yOutput存的是预测输出后的1*10的向量结果
			row := rowT.Data().([]float64)
			var rowToWrite []string

			for k := 0; k < 10; k++ {
				rowToWrite = append(rowToWrite, strconv.FormatFloat(row[k], 'f', 6, 64))
			}
			matrixToWrite = append(matrixToWrite, rowToWrite)
		}

		csvWriter := csv.NewWriter(file)
		csvWriter.WriteAll(matrixToWrite)
		csvWriter.Flush()

		vm.Reset()
		bar.Increment()
	}
	log.Printf("Epoch Test | cost %v", costVal)

}
