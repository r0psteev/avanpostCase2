package main

import (
	"os"
	"io"
	"image"
	"fmt"
	"log"
	"sort"
	"strings"
	"bufio"
	"runtime"
	"sync"
	"time"

	"golang.org/x/image/bmp"
	"github.com/disintegration/imaging"
)

var (

	trainDataset,  testDataset =  `../Датасет/Датасет/SOCOFing/Real/`, `../Датасет/Датасет/SOCOFing/Altered/Altered-Hard/`
	//trainDataset,  testDataset =  ``, ``
	model_cache_file = `./model.cache.txt`
	model_predictions_file = `./model.predictions.txt`
	digestLen = 25
	nNcpu = runtime.NumCPU()
)

func main() {

	if len(os.Args) < 2 {
		log.Printf("Usage: %s [-test|-train]", os.Args[0])
		log.Printf("%s -train <directory_of_training_images>", os.Args[0])
		log.Printf("%s -test <directory_of_images_to_test>", os.Args[0])
		return
	}

	if os.Args[1] == "-train" {
		if len(os.Args) > 2 {
			trainDataset = os.Args[2]
		}
		files, err := os.ReadDir(trainDataset);
		if err != nil {
			panic(err)
		}
		fileList := []string{}
		for _, file := range files {
			fileList = append(fileList, file.Name())
		}
		Train(fileList)
	}else if os.Args[1] == "-test" {
		
		/*
		if len(os.Args) > 2 {
			testDataset = os.Args[2]
		}
		files, err := os.ReadDir(testDataset);
		if err != nil {
			panic(err)
		}


		fileList := []string{}
		for _, file := range files {
			fileList = append(fileList, file.Name())
		}


		Test(fileList)*/
		Test()
	}
}


// 1. INPUT : All the image files in direcotry
// 2. OUTPUT : A text file `model.cache.txt` which contains all the generated digests for the images
// 3. Candidate for concurrency at every file iteration.
func Train(fileList []string) {

	digestsCache := []float64{}
	digestToSubjectID := make(map[float64]string) // map digest to Subject id

	log.Println("[!] Starting Training")
	for _, fileName := range fileList {
		// 1. load io.Reader for image file from filesystem
		filepath := fmt.Sprintf(`%s/%s`, trainDataset, fileName)
		img, err := loadImageFile(filepath)
		if err != nil {
			panic(err)
		}
		

		// 2. Convert normal image into GrayScale image.
		grayImg, err := toGrayScale(img); 
		if err != nil {
			panic(err)
		}
	
		// 3. Apply `Sobel Operator` Horizontal kernel on image matrix
		sobelImg := ModelSobel(grayImg)
		sobelImgGray, err := toGrayScale(sobelImg)
		if err != nil {
			panic(err)
		}
		
		top_pixel_values, top_frequencies := PixelFrequencyDistribution(sobelImgGray.Pix)
		
		digest := digestFrequencyDistribution(top_pixel_values, top_frequencies)
		digestsCache = append(digestsCache, digest)
		subjectEntry := strings.Split(fileName, "_")
		digestToSubjectID[digest] = subjectEntry[0]
	}

	sort.SliceStable(digestsCache, func(i, j int) bool { return digestsCache[i] < digestsCache[j]})
	
	// save the values to a file.
	f, err := os.Create(model_cache_file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	log.Println("[+] Ended Training")
	log.Println("[!] Saving computed parameters to disk")
	for _, digest := range digestsCache {
		_, err := fmt.Fprintf(f, "%f:%s\n", digest, digestToSubjectID[digest])
		if err != nil {
			panic(err)
		}
	}
	log.Printf("[+] Parameters saved to %s\n", model_cache_file)
}


// 1. INPUT : An image
// 2. OUTPUT: The person ID associated to that file.
func Test() {

	// test data

	test_data := map[string]string{"00000.bmp": "64__M_Right_index_finger", "00001.bmp": "452__F_Left_index_finger", "00002.bmp": "351__M_Left_little_finger", "00003.bmp": "421__F_Right_index_finger", "00004.bmp": "540__F_Right_ring_finger", "00005.bmp": "410__M_Right_thumb_finger", "00006.bmp": "586__M_Left_thumb_finger", "00007.bmp": "75__F_Right_ring_finger", "00008.bmp": "177__F_Left_ring_finger", "00009.bmp": "365__M_Left_middle_finger", "00010.bmp": "312__M_Right_little_finger", "00011.bmp": "575__M_Right_index_finger", "00012.bmp": "267__M_Left_thumb_finger", "00013.bmp": "79__M_Right_middle_finger", "00014.bmp": "122__M_Left_index_finger", "00015.bmp": "218__M_Left_middle_finger", "00016.bmp": "25__F_Left_little_finger", "00017.bmp": "136__F_Right_little_finger", "00018.bmp": "115__F_Right_middle_finger", "00019.bmp": "558__M_Right_little_finger", "00020.bmp": "249__M_Left_thumb_finger", "00021.bmp": "281__M_Left_index_finger", "00022.bmp": "391__M_Right_little_finger", "00023.bmp": "211__M_Right_thumb_finger", "00024.bmp": "451__M_Right_little_finger", "00025.bmp": "453__F_Left_ring_finger", "00026.bmp": "334__F_Right_ring_finger", "00027.bmp": "149__F_Right_little_finger", "00028.bmp": "306__M_Left_little_finger", "00029.bmp": "77__M_Left_thumb_finger", "00030.bmp": "78__F_Right_middle_finger", "00031.bmp": "389__F_Right_middle_finger", "00032.bmp": "119__F_Left_thumb_finger", "00033.bmp": "468__F_Right_little_finger", "00034.bmp": "52__M_Left_little_finger", "00035.bmp": "217__M_Right_ring_finger", "00036.bmp": "294__M_Left_middle_finger", "00037.bmp": "215__M_Right_little_finger", "00038.bmp": "312__M_Left_thumb_finger", "00039.bmp": "372__M_Left_middle_finger", "00040.bmp": "276__M_Left_little_finger", "00041.bmp": "53__M_Right_thumb_finger", "00042.bmp": "378__F_Left_middle_finger", "00043.bmp": "175__M_Right_index_finger", "00044.bmp": "130__F_Left_thumb_finger", "00045.bmp": "411__M_Right_thumb_finger", "00046.bmp": "475__M_Left_index_finger", "00047.bmp": "88__F_Left_middle_finger", "00048.bmp": "142__F_Left_middle_finger", "00049.bmp": "309__M_Right_little_finger", "00050.bmp": "460__M_Left_middle_finger", "00051.bmp": "428__M_Right_little_finger", "00052.bmp": "563__M_Right_index_finger", "00053.bmp": "476__M_Left_middle_finger", "00054.bmp": "59__F_Right_thumb_finger", "00055.bmp": "125__M_Right_middle_finger", "00056.bmp": "396__M_Left_little_finger", "00057.bmp": "219__M_Left_index_finger", "00058.bmp": "413__M_Left_middle_finger", "00059.bmp": "179__M_Left_little_finger", "00060.bmp": "110__F_Left_thumb_finger", "00061.bmp": "333__M_Left_index_finger", "00062.bmp": "311__M_Right_index_finger", "00063.bmp": "290__M_Left_thumb_finger", "00064.bmp": "330__M_Right_middle_finger", "00065.bmp": "442__F_Right_ring_finger", "00066.bmp": "446__M_Right_index_finger", "00067.bmp": "278__M_Right_little_finger", "00068.bmp": "233__M_Right_ring_finger", "00069.bmp": "205__F_Left_thumb_finger", "00070.bmp": "431__M_Left_little_finger", "00071.bmp": "581__F_Right_middle_finger", "00072.bmp": "300__F_Right_index_finger", "00073.bmp": "354__M_Left_middle_finger", "00074.bmp": "426__M_Left_ring_finger", "00075.bmp": "481__F_Left_thumb_finger", "00076.bmp": "172__M_Right_little_finger", "00077.bmp": "407__M_Left_index_finger", "00078.bmp": "481__F_Left_little_finger", "00079.bmp": "468__F_Right_middle_finger", "00080.bmp": "518__M_Right_thumb_finger", "00081.bmp": "274__M_Right_ring_finger", "00082.bmp": "263__F_Right_thumb_finger", "00083.bmp": "120__M_Right_index_finger", "00084.bmp": "481__F_Right_little_finger", "00085.bmp": "391__M_Right_index_finger", "00086.bmp": "518__M_Right_middle_finger", "00087.bmp": "129__M_Left_little_finger", "00088.bmp": "318__F_Left_index_finger", "00089.bmp": "577__M_Left_middle_finger", "00090.bmp": "212__M_Left_ring_finger", "00091.bmp": "304__M_Left_index_finger", "00092.bmp": "158__M_Right_little_finger", "00093.bmp": "361__M_Left_middle_finger", "00094.bmp": "239__M_Right_little_finger", "00095.bmp": "487__M_Left_middle_finger", "00096.bmp": "294__M_Right_little_finger", "00097.bmp": "30__F_Left_index_finger", "00098.bmp": "560__F_Right_little_finger", "00099.bmp": "93__M_Left_ring_finger", "00100.bmp": "182__M_Left_little_finger", "00101.bmp": "587__M_Left_ring_finger", "00102.bmp": "518__M_Left_index_finger", "00103.bmp": "235__M_Right_middle_finger", "00104.bmp": "391__M_Left_ring_finger", "00105.bmp": "504__M_Left_thumb_finger", "00106.bmp": "600__M_Right_index_finger", "00107.bmp": "114__F_Right_ring_finger", "00108.bmp": "477__M_Right_thumb_finger", "00109.bmp": "525__M_Left_middle_finger", "00110.bmp": "154__F_Right_little_finger", "00111.bmp": "117__F_Right_little_finger", "00112.bmp": "97__M_Left_ring_finger", "00113.bmp": "221__M_Right_little_finger", "00114.bmp": "174__F_Left_ring_finger", "00115.bmp": "106__M_Left_middle_finger", "00116.bmp": "466__F_Left_ring_finger", "00117.bmp": "147__M_Left_ring_finger", "00118.bmp": "273__M_Left_middle_finger", "00119.bmp": "465__F_Left_middle_finger", "00120.bmp": "165__M_Left_ring_finger", "00121.bmp": "35__M_Left_thumb_finger", "00122.bmp": "494__F_Left_ring_finger", "00123.bmp": "472__M_Left_ring_finger", "00124.bmp": "105__M_Right_middle_finger", "00125.bmp": "456__M_Right_middle_finger", "00126.bmp": "70__M_Right_middle_finger", "00127.bmp": "399__M_Right_ring_finger", "00128.bmp": "270__M_Right_thumb_finger", "00129.bmp": "196__M_Right_little_finger", "00130.bmp": "110__F_Right_thumb_finger", "00131.bmp": "126__F_Right_index_finger", "00132.bmp": "500__M_Right_middle_finger", "00133.bmp": "171__M_Left_little_finger", "00134.bmp": "55__M_Left_ring_finger", "00135.bmp": "407__M_Left_little_finger", "00136.bmp": "533__M_Left_thumb_finger", "00137.bmp": "562__F_Left_thumb_finger", "00138.bmp": "238__M_Left_ring_finger", "00139.bmp": "245__M_Left_ring_finger", "00140.bmp": "284__M_Left_thumb_finger", "00141.bmp": "261__M_Right_middle_finger", "00142.bmp": "217__M_Right_thumb_finger", "00143.bmp": "64__M_Left_thumb_finger", "00144.bmp": "362__M_Left_little_finger", "00145.bmp": "121__F_Left_little_finger", "00146.bmp": "435__F_Left_thumb_finger", "00147.bmp": "416__M_Right_middle_finger", "00148.bmp": "308__M_Right_middle_finger", "00149.bmp": "225__M_Left_little_finger", "00150.bmp": "347__M_Left_thumb_finger", "00151.bmp": "313__M_Left_index_finger", "00152.bmp": "396__M_Left_ring_finger", "00153.bmp": "52__M_Right_middle_finger", "00154.bmp": "514__F_Right_little_finger", "00155.bmp": "254__M_Left_ring_finger", "00156.bmp": "354__M_Right_thumb_finger", "00157.bmp": "519__M_Left_middle_finger", "00158.bmp": "132__M_Left_index_finger", "00159.bmp": "524__M_Right_little_finger", "00160.bmp": "191__F_Right_middle_finger", "00161.bmp": "352__M_Left_ring_finger", "00162.bmp": "368__M_Left_middle_finger", "00163.bmp": "264__M_Right_thumb_finger", "00164.bmp": "73__M_Right_ring_finger", "00165.bmp": "221__M_Left_ring_finger", "00166.bmp": "104__M_Left_index_finger", "00167.bmp": "367__M_Right_ring_finger", "00168.bmp": "229__M_Right_index_finger", "00169.bmp": "374__M_Right_middle_finger", "00170.bmp": "23__M_Right_index_finger", "00171.bmp": "342__M_Left_ring_finger", "00172.bmp": "597__M_Right_middle_finger", "00173.bmp": "401__M_Right_little_finger", "00174.bmp": "321__M_Right_thumb_finger", "00175.bmp": "266__M_Left_index_finger", "00176.bmp": "594__M_Right_thumb_finger", "00177.bmp": "286__M_Right_little_finger", "00178.bmp": "139__M_Right_middle_finger", "00179.bmp": "479__F_Left_thumb_finger", "00180.bmp": "66__F_Left_index_finger", "00181.bmp": "15__F_Left_index_finger", "00182.bmp": "503__M_Left_little_finger", "00183.bmp": "30__F_Left_little_finger", "00184.bmp": "469__M_Left_little_finger", "00185.bmp": "534__F_Left_ring_finger", "00186.bmp": "314__M_Left_little_finger", "00187.bmp": "519__M_Right_index_finger", "00188.bmp": "250__F_Left_middle_finger", "00189.bmp": "13__F_Left_thumb_finger", "00190.bmp": "203__M_Left_index_finger", "00191.bmp": "13__F_Right_index_finger", "00192.bmp": "122__M_Left_ring_finger", "00193.bmp": "493__M_Right_thumb_finger", "00194.bmp": "409__M_Right_little_finger", "00195.bmp": "86__M_Right_little_finger", "00196.bmp": "521__M_Right_index_finger", "00197.bmp": "165__M_Right_middle_finger", "00198.bmp": "447__M_Left_middle_finger", "00199.bmp": "366__M_Left_middle_finger", "00200.bmp": "180__F_Right_ring_finger", "00201.bmp": "112__M_Left_middle_finger", "00202.bmp": "50__M_Left_little_finger", "00203.bmp": "451__M_Right_index_finger", "00204.bmp": "77__M_Right_ring_finger", "00205.bmp": "36__M_Right_middle_finger", "00206.bmp": "374__M_Left_thumb_finger", "00207.bmp": "554__M_Left_little_finger", "00208.bmp": "252__F_Right_middle_finger", "00209.bmp": "379__F_Right_little_finger", "00210.bmp": "421__F_Left_thumb_finger", "00211.bmp": "235__M_Left_little_finger", "00212.bmp": "467__M_Right_middle_finger", "00213.bmp": "141__F_Right_ring_finger", "00214.bmp": "7__M_Left_little_finger", "00215.bmp": "197__M_Right_ring_finger", "00216.bmp": "583__M_Left_index_finger", "00217.bmp": "408__M_Left_little_finger", "00218.bmp": "167__M_Left_little_finger", "00219.bmp": "84__M_Left_thumb_finger", "00220.bmp": "597__M_Left_ring_finger", "00221.bmp": "341__M_Left_index_finger", "00222.bmp": "390__F_Right_thumb_finger", "00223.bmp": "37__M_Right_middle_finger", "00224.bmp": "40__F_Left_index_finger", "00225.bmp": "163__M_Left_thumb_finger", "00226.bmp": "394__M_Left_index_finger", "00227.bmp": "54__M_Right_index_finger", "00228.bmp": "202__M_Left_thumb_finger", "00229.bmp": "517__M_Right_thumb_finger", "00230.bmp": "421__F_Left_little_finger", "00231.bmp": "496__M_Left_thumb_finger", "00232.bmp": "174__F_Right_little_finger", "00233.bmp": "139__M_Right_thumb_finger", "00234.bmp": "253__F_Right_index_finger", "00235.bmp": "90__M_Left_index_finger", "00236.bmp": "46__M_Left_index_finger", "00237.bmp": "122__M_Left_thumb_finger", "00238.bmp": "313__M_Left_middle_finger", "00239.bmp": "173__F_Right_thumb_finger", "00240.bmp": "325__M_Right_index_finger", "00241.bmp": "89__M_Right_middle_finger", "00242.bmp": "90__M_Right_middle_finger", "00243.bmp": "337__F_Right_ring_finger", "00244.bmp": "374__M_Right_index_finger", "00245.bmp": "504__M_Right_index_finger", "00246.bmp": "300__F_Right_little_finger", "00247.bmp": "596__M_Left_ring_finger", "00248.bmp": "336__M_Right_little_finger", "00249.bmp": "47__F_Right_thumb_finger", "00250.bmp": "456__M_Left_middle_finger", "00251.bmp": "114__F_Left_ring_finger", "00252.bmp": "258__M_Right_middle_finger", "00253.bmp": "564__M_Left_thumb_finger", "00254.bmp": "445__M_Left_thumb_finger", "00255.bmp": "359__M_Left_index_finger", "00256.bmp": "196__M_Right_middle_finger", "00257.bmp": "209__F_Right_index_finger", "00258.bmp": "228__M_Left_little_finger", "00259.bmp": "265__M_Left_ring_finger", "00260.bmp": "444__M_Right_little_finger", "00261.bmp": "462__M_Left_little_finger", "00262.bmp": "167__M_Right_ring_finger", "00263.bmp": "315__F_Left_middle_finger", "00264.bmp": "427__M_Left_index_finger", "00265.bmp": "98__M_Left_little_finger", "00266.bmp": "593__M_Right_ring_finger", "00267.bmp": "571__F_Left_thumb_finger", "00268.bmp": "504__M_Right_ring_finger", "00269.bmp": "206__M_Left_ring_finger", "00270.bmp": "382__M_Right_thumb_finger", "00271.bmp": "108__M_Left_ring_finger", "00272.bmp": "474__M_Right_thumb_finger", "00273.bmp": "250__F_Right_index_finger", "00274.bmp": "570__M_Left_little_finger", "00275.bmp": "211__M_Right_ring_finger", "00276.bmp": "62__M_Right_middle_finger", "00277.bmp": "420__M_Right_middle_finger", "00278.bmp": "151__M_Right_index_finger", "00279.bmp": "421__F_Right_middle_finger", "00280.bmp": "403__M_Right_little_finger", "00281.bmp": "130__F_Left_little_finger", "00282.bmp": "585__M_Right_ring_finger", "00283.bmp": "127__F_Left_index_finger", "00284.bmp": "507__M_Left_middle_finger", "00285.bmp": "480__M_Left_little_finger", "00286.bmp": "106__M_Right_middle_finger", "00287.bmp": "377__M_Right_little_finger", "00288.bmp": "586__M_Right_little_finger", "00289.bmp": "45__M_Left_index_finger", "00290.bmp": "484__M_Left_ring_finger", "00291.bmp": "261__M_Right_little_finger", "00292.bmp": "514__F_Left_ring_finger", "00293.bmp": "22__M_Right_little_finger", "00294.bmp": "507__M_Right_thumb_finger", "00295.bmp": "359__M_Right_thumb_finger", "00296.bmp": "475__M_Right_little_finger", "00297.bmp": "479__F_Left_index_finger", "00298.bmp": "72__M_Right_little_finger", "00299.bmp": "401__M_Left_middle_finger"}

	// load model.cache.txt
	f, err := os.Open(model_cache_file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	digestsCache := []float64{}
	digestToSubjectID := make(map[float64]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		digestStr := strings.Split(line, ":")[0]
		subjectId := strings.Split(line, ":")[1]
		var digest float64
		_, err := fmt.Sscanf(digestStr, "%f", &digest)
		if err != nil {
			panic(err)
		}
		digestsCache = append(digestsCache, digest)
		digestToSubjectID[digest] = subjectId
	}



	log.Printf("Begining Testing with %d cores\n", nNcpu)
	startTime := time.Now()

	wg := sync.WaitGroup{}
	fileNamesChannel := make(chan string, nNcpu)
	realSubjectIdChannel := make(chan string, nNcpu)
	resultsChannel := make(chan string, nNcpu)
	for i:=0; i<nNcpu; i++ {

		wg.Add(1)

		go func(){
			defer wg.Done()

			for fileName := range fileNamesChannel {

				// 1. load io.Reader for image file from filesystem
				filepath := fmt.Sprintf(`%s`, fileName)
				img, err := loadImageFile(filepath)
				if err != nil {
					panic(err)
				}
				

				// 2. Convert normal image into GrayScale image.
				grayImg, err := toGrayScale(img); 
				if err != nil {
					panic(err)
				}
			
				// 3. Apply `Sobel Operator` Horizontal kernel on image matrix
				sobelImg := ModelSobel(grayImg)
				sobelImgGray, err := toGrayScale(sobelImg)
				if err != nil {
					panic(err)
				}
				
				top_pixel_values, top_frequencies := PixelFrequencyDistribution(sobelImgGray.Pix)
				
				digest := digestFrequencyDistribution(top_pixel_values, top_frequencies)

				index, _ := Search(digestsCache, digest)
				
				var predictedSubjectId string
				predictedSubjectId = digestToSubjectID[digestsCache[index]]
				resultsChannel <- fmt.Sprintf("%s:%s", predictedSubjectId, <-realSubjectIdChannel)
			}
		}()
	}


	fp, _ := os.Create(model_predictions_file)

	// testing the list
	/*
	for _, fileName := range fileList {
		fileNamesChannel <- fileName
		fmt.Fprintln(fp, <- resultsChannel)
	}*/

	for image,  label := range test_data {
		fileNamesChannel <- fmt.Sprintf("./test/images/%s", image)
		realSubjectIdChannel <- label
		fmt.Fprintln(fp, <- resultsChannel)
	}

	log.Println("Tests ended")
	log.Println("Duration := ", time.Now().Sub(startTime))


	log.Println("Accuray:")
	pass, total := Accuracy()
	log.Printf("Total samples = %d, Pass := %d/%d,  Failed := %d/%d\n", total, pass, total, total-pass, total)
	

}

// O(n)
func Search(digestsCache []float64, digest float64) (int, int) {
	if digest > digestsCache[len(digestsCache)-1] {
		return len(digestsCache)-1, int(digest - digestsCache[len(digestsCache)-1])
	}
	if digest < digestsCache[0] {
		return 0, int(digestsCache[0] - digest)
	}
	for i, v := range digestsCache{
		if v == digest {
			return i, int(v)-int(digest)
		}
		if v > digest {
			if i>0 && (digest - digestsCache[i-1]) < (v-digest) {
				return i-1, int(digest)-int(digestsCache[i-1])
			}
			return i, int(v)-int(digest)
		}
	}
	panic("unreachable")
}

func Accuracy() (pass int, total int) {

	f, err := os.Open(model_predictions_file)
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		predictedSubjectId := strings.Split(line, ":")[0]
		fileName := strings.Split(line, ":")[1]
		subjectId := strings.Split(fileName, "_")[0]

		if subjectId == predictedSubjectId {
			pass += 1
		}
		total += 1
	}
	return 
}

func loadImageFile(filepath string) (image.Image, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, err := readBMPImage(file)
	if err != nil {
		panic(err)
	}

	return img, nil
}

func readBMPImage(r io.Reader) (image.Image, error) {
	img, err := bmp.Decode(r)
	if err != nil {
		return nil, err
	}
	return img, nil
}

// https://riptutorial.com/go/example/31693/convert-color-image-to-grayscale
func toGrayScale(img image.Image) (*image.Gray, error) {
	grayImg := image.NewGray(img.Bounds())
	for row := img.Bounds().Min.Y; row < img.Bounds().Max.Y; row ++ {
		for col := img.Bounds().Min.X; col < img.Bounds().Max.X; col++ {
			grayImg.Set(col, row, img.At(col, row))
		}
	}
	return grayImg, nil
}

func saveImageFile(filepath string, img image.Image) (error) {
	f, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := bmp.Encode(f, img); err != nil {
		return err
	}
	return nil
}

//https://www.geeksforgeeks.org/image-edge-detection-operators-in-digital-image-processing/
func ModelSobel(img image.Image) (*image.NRGBA){

	/*
	kernel := [9]float64{
		1, 2, 1,
		0, 0, 0,
		-1, -1, -1,
	}
	*/

	/*
	// 100% on Alter-Medium
	// 90% on Real
	// very poor on Alter-Easy and Alter-Hard (Unseen)
	kernel := [9]float64 {
		+3, +10, +3,
		0, 0, 0,
		-3, -10, -3,
	}*/

	kernel := [25]float64 {
	 	2, 2, 4, 2, 2,
	 	1, 1, 2, 1, 1,
	 	0, 0, 0, 0, 0,
	 	-1, -1, -2, -1, -1,
	 	-2, -2, -4, -2, -2,
	 }

	/*
	kernel := [25]float64 {
		2, 1, 0, -1, -2,
		2, 1, 0, -1, -2,
		4, 2, 0, -2, -4,
		2, 1, 0, -1, -2,
		2, 1, 0, -1, -2,
	}*/

	/*
	kernel := [9]float64{
		1, 0, -1,
		2, 0, -2,
		1, 0, -1,
	} */

	imgConvo := imaging.Convolve5x5(
		img,
		kernel,
		nil,
	)

	return imgConvo
}

// The idea is to find the most popular pixels within the Edged image
// and their respective frequencies.
// Ratios between their frequencies are likely going to be the same accross
// images showing the same fingeprint.
// kind of identification based on frequency of pixels
func PixelFrequencyDistribution(pixels []uint8) ([]uint, []uint){
	var arr  [256]uint // pixel values range from 0-255
	var top_pixel_values []uint // the top most common pixel values execept '0'
	var top_frequencies []uint // the frequencies of the top

	for i:=0; i<len(pixels); i++ {
		if pixels[i] != 0 {
			arr[pixels[i]] += 1
		}
	}

	
	for i:=0; i<digestLen; i++ {
		pixel, frequency := findMaxElement(arr)
		top_pixel_values = append(top_pixel_values, pixel)
		top_frequencies = append(top_frequencies, frequency)
		arr[top_pixel_values[i]] = 0
	}

	return top_pixel_values, top_frequencies
}

func findMaxElement(arr [256]uint) (uint, uint) {
	var max uint = 0;
	var position uint = 0;
	for i:=0; i<len(arr); i++ {
		if arr[i] >= max {
			max = arr[i]
			position = uint(i)
		}
	}
	return position, max
}

// just a simple attempt to combine the frequencies of all the top5 elements
// into a searchable unique integer.
func digestFrequencyDistribution(top_pixel_values, top_frequencies []uint) float64 {
	var digest float64 = 1
	for i:=0; i<len(top_frequencies); i++ {
		//digest = digest*padding(top_frequencies[i]) + top_frequencies[i]
		digest =  digest * (3 + float64(top_pixel_values[i])/float64(top_frequencies[i]))
	}
	return digest
}
//https://stackoverflow.com/questions/28029518/golang-combine-two-numbers
func padding(n uint) uint {
	var  p  uint = 1
	for p<n {
		p *= 10
	}
	return p
}