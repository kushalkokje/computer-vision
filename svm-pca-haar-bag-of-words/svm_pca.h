#include <fstream>
#include <sstream>

class SVM_PCA : public Classifier {
 private:
  // Number of support vectors
  int numSVs;
  bool isColor;

  // Variable to hold image size
  int xSize;

  // File names
  string trainData;
  string model;
  string testData;
  string predictions;

  // Class names
  vector<string> classList;

  // Class name to number mapping
  map<string, int> classIndexMap;

 public:
  ofstream outfile;
  SVM_PCA(const vector<string> &_class_list, bool _isColor, int _numSVs)
      : Classifier(_class_list) {
    isColor = _isColor;
    // Update image size based on color channels
    if (isColor) {
      xSize = size * size * 3;
    } else {
      xSize = size * size;
    }
    numImages = 0;
    numSVs = _numSVs;
    classList = _class_list;
    // Update class mapping
    for (int i = 0; i < _class_list.size(); i++) {
      classIndexMap[_class_list[i]] = (i + 1);
    }
  }

  void convertTosvmformat(const Dataset &filenames, string outFileName) {
    outfile.open(outFileName.c_str());
    int k = 0;
    vector<int> nImages;
    for (Dataset::const_iterator c_iter = filenames.begin();
         c_iter != filenames.end(); ++c_iter) {
      nImages.push_back(c_iter->second.size());
    }

    // Training matrix
    // 400 x 1250
    CImg<double> train_data(xSize, getSize(filenames), 1);
    // k x 1250
    CImg<double> train_data_pc(numSVs, getSize(filenames), 1);

    // Got Principal components 400 x 400 (int case of 20x20 image)
    CImg<double> pcList = train_pca(filenames, train_data);
    // train_data.display();
    cout << "PC dimensions " << pcList.height() << " x " << pcList.width()
         << endl;

    // Extract top k principle components
    // k x 400
    CImg<double> topKPC(numSVs, pcList.height());

    // Store principle components so we can reuse while testing
    for (int tpwIndex = 0; tpwIndex < topKPC.width(); tpwIndex++) {
       CImg<double> pcImage(1, pcList.height());

      for (int tphIndex = 0; tphIndex < topKPC.height(); tphIndex++) {
        topKPC(tpwIndex, tphIndex) = pcList(tpwIndex, tphIndex);
        pcImage(1, tphIndex) = topKPC(tpwIndex, tphIndex);
	    }
      stringstream ss;
      ss << tpwIndex+1;
      string str = ss.str();
      string imgName = "intermediate/PC/PC_"+str+".png";
      pcImage.resize(size,size,1).normalize(0,255).save(imgName.c_str()); 
      
	   }

     //Save top k principle components
    topKPC.normalize(0, 255).save("intermediate/topk_img.png");

    // Transform train data as per principle components
    train_data_pc = dot(topKPC, train_data);
    train_data_pc.print();

    //Write to SVM format file
    int imgIndex = 0;
    for (Dataset::const_iterator c_iter = filenames.begin();
         c_iter != filenames.end(); ++c_iter) {
      for (int i = 0; i < c_iter->second.size(); i++) {
        int cIndex = getClassIndex(c_iter->second[i], classIndexMap);
        outfile << cIndex << " ";

        for (int j = 0; j < numSVs; j++)
          outfile << j + 1 << ":" << train_data_pc(j, imgIndex) << " ";
        imgIndex++;

        outfile << endl;
      }
    }

    outfile.close();
  }

  // SVM training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a
  // file
  virtual void train(const Dataset &filenames) {
    if (isColor) {
      trainData = "SVMtrain/train_svm_color_eigen.dat";
      model = "SVMmodels/svm_color_model_eigen";
      convertTosvmformat(filenames, trainData);
      string svmCommand =
          "./svm_multiclass_learn -c 1 " + trainData + " " + model;
      system(svmCommand.c_str());
    } else {
      trainData = "SVMtrain/train_svm_grey_eigen.dat";
      model = "SVMmodels/svm_grey_model_eigen";
      convertTosvmformat(filenames, trainData);
      string svmCommand =
          "./svm_multiclass_learn -c 1 " + trainData + " " + model;
      system(svmCommand.c_str());
    }
  }

  virtual string classify(const string &filename) {
    int predClassIndex;

    if (isColor) {
      testData = "SVMtest/test_svm_color_eigen.dat";
      model = "SVMmodels/svm_color_model_eigen";
      predictions = "SVMpreds/preds_svm_color_eigen.dat";
    } else {
      testData = "SVMtest/test_svm_grey.dat";
      model = "SVMmodels/svm_grey_model_eigen";
      predictions = "SVMpreds/preds_svm_grey_eigen.dat";
    }

    // kx400 in case of 20x20 image
    //Read intermediate principle components
    CImg<double> topK("intermediate/topk_img.png");
    topK.print();

    outfile.open(testData.c_str());
    
    //Write desired class
    outfile << getClassIndex(filename, classIndexMap) << " ";
    CImg<double> iFeatures = extract_features(filename, isColor);

    //Transform test data
    CImg<double> testFeatures = dot(topK, iFeatures);

    for (int fIndex = 0; fIndex < testFeatures.size(); fIndex++) {
      outfile << " " << fIndex + 1 << ":" << testFeatures[fIndex];
    }

    outfile << endl;
    cout << "features written" << endl;

    outfile.close();
    string svmCommand = "./svm_multiclass_classify " + testData + " " + model +
                        " " + predictions;
    system(svmCommand.c_str());
    // Read predictions
    // Parse output file
    ifstream preds(predictions.c_str());

    // Convert to integer
    preds>>predClassIndex;

    // Return class name
    return classList[predClassIndex - 1];
  }

  // void classify(const Dataset &filenames)
  // {
  //   if(isColor){
  //     outFileName = "test_svm4800_eigen.dat";
  //     convertTosvmformat(filenames,outFileName);
  //     system("./svm_multiclass_classify test_svm4800_eigen.dat
  //     svm4800model_eigen test_preds4800_eigen");
  //   }else{
  //     outFileName = "test_svm1600_eigen.dat";
  //     convertTosvmformat(filenames,outFileName);
  //     system("./svm_multiclass_classify test_svm1600_eigen.dat
  //     svm1600model_eigen test_preds1600_eigen");
  //   }
  // }

  virtual void load_model() {
    // DO nothing
  }

 protected:
  // extract features from an image, which in this case just involves resampling
  // and
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename, bool isColor) {
    CImg<double> a;
    if (isColor)
      a = CImg<double>(filename.c_str()).resize(size, size, 1, 3).unroll('x');
    else
      a = CImg<double>(filename.c_str()).resize(size, size, 1, 1).unroll('x');

    // cout << a.width() << a.height() << a.depth() << endl;
    return a;
  }

  CImg<double> train_pca(const Dataset &filenames,
                         CImg<double> &class_vectors) {
    int j = 0;
    CImg<double> eigenVal;
    CImg<double> eigenVec;

    for (Dataset::const_iterator c_iter = filenames.begin();
         c_iter != filenames.end(); ++c_iter) {
      cout << "Processing folder " << c_iter->first << endl;

      for (int i = 0; i < c_iter->second.size(); i++) {
        class_vectors = class_vectors.draw_image(
            0, j, 0, 0, extract_features(c_iter->second[i].c_str(), isColor));
        j++;
      }
    }

    cout << "Calculating covariance matrix" << endl;
    CImg<double> coVarMat = dot(class_vectors, class_vectors.get_transpose());

    cout << "Calculating eigenvalues and eigenvectors" << endl;
    coVarMat.symmetric_eigen(eigenVal, eigenVec);

    cout << "Top 20 eigen values ";
    for (int eIndex = 0; eIndex < 10; eIndex++) {
      cout << eigenVal[eIndex] << " ";
    }
    cout << endl;

    cout << "Last 20 eigen values ";
    for (int eIndex = eigenVal.size(); eIndex > eigenVal.size() - 10;
         eIndex--) {
      cout << eigenVal[eIndex] << " ";
    }
    cout << endl;

    return eigenVec;
  }

  int getSize(const Dataset &filenames) {
    for (Dataset::const_iterator c_iter = filenames.begin();
         c_iter != filenames.end(); ++c_iter) {
      for (int i = 0; i < c_iter->second.size(); i++) numImages++;
    }

    return numImages;
  }

  CImg<double> dot(CImg<double> A, CImg<double> B) {
    CImg<double> H(A.width(), B.height());
    cout << "Dot dimensions" << H.width() << " " << H.height() << endl;
    for (int x = 0; x < H.width(); x++) {
      for (int y = 0; y < H.height(); y++) {
        H(x, y) = A.get_column(x).dot(B.get_row(y));
      }
    }

    return H;
  }

  static const int size = 20;  // subsampled image resolution
  string modelName;
  int numImages;
};
