#include <fstream>

// SVM Class
class SVM : public Classifier
{
private: 
  bool isColor;
  string trainData;
  string model;
  string testData;
  string predictions;
  vector<string> classList;
  map<string,int> classIndex;

public:
  ofstream outfile;
  SVM(const vector<string> &_class_list,bool _isColor) : Classifier(_class_list) {
      isColor = _isColor;
      classList = _class_list;
	for(int i=0;i<_class_list.size();i++){
      	classIndex[_class_list[i]] = (i+1);
      }
  }


  void convertTosvmformat(const Dataset &filenames,string outFileName){
    outfile.open(outFileName.c_str());  
    int k = 0;
    int xSize = 0; 
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
    {
      //If color image is used then size will be 3 times than grey scale image
      if(isColor)
        xSize = size*size*3;
      else
        xSize = size*size;

      CImg<double> class_vectors(xSize, c_iter->second.size(), 1);

      cout << "Processing folder: " << c_iter->first << endl;

      // convert each image to be a row of this "model" image
      for(int i=0; i<c_iter->second.size(); i++)
        class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str(),isColor));

      //cout << "class vectors width" << class_vectors.width() << "height " << class_vectors.height() <<endl;
      // Increment class value as SVM supports from 1
      k = k + 1;

      // Write in SVM format
      for(int imgIndex=0; imgIndex<class_vectors.height() ; imgIndex++) { 

        outfile<<k<<" ";
        for(int j=0; j<class_vectors.width() ;j++) {

          outfile<<j+1<<":"<<class_vectors(j,imgIndex)<<" "; 
        }

        outfile<<endl;  
      }
    }
      
    outfile.close();
  }
  
  // SVM training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames) 
  {
    if(isColor){
      trainData = "SVMtrain/train_svm_color.dat";
      model = "SVMmodels/svm_color_model";
      convertTosvmformat(filenames,trainData);
      string svmCommand = "./svm_multiclass_learn -c 1 " + trainData+ " " + model;
      system(svmCommand.c_str());
    }else{
      trainData = "SVMtrain/train_svm_grey.dat";
      model = "SVMmodels/svm_grey_model";
      convertTosvmformat(filenames,trainData);
      string svmCommand = "./svm_multiclass_learn -c 1 " + trainData+ " " + model;
      system(svmCommand.c_str());
    }
  }

  virtual string classify(const string &filename){
	//We will hold predicted value in this variable
	int predClassIndex;

	if(isColor){
		testData = "SVMtest/test_svm_color.dat";
		model = "SVMmodels/svm_color_model";
		predictions = "SVMpreds/preds_svm_color.dat";
	}
	else{
		testData = "SVMtest/test_svm_grey.dat";
		model = "SVMmodels/svm_grey_model";
		predictions = "SVMpreds/preds_svm_grey.dat";
	}
	CImg<double> testFeatures = extract_features(filename,isColor);
    
	// Write to SVM format
  outfile.open(testData.c_str()); 

  //Get class index based on folder name 
	outfile << getClassIndex(filename,classIndex) << " ";

	for(int fIndex = 0; fIndex<testFeatures.size();fIndex++){
		outfile<<" "<<fIndex+1<<":"<<testFeatures[fIndex];
	}

	outfile<<endl;
   	//cout << "features written" << endl;

	outfile.close();

	//SVM command
	string svmCommand = "./svm_multiclass_classify "+ testData+ " " +model+ " "+predictions; 
	system(svmCommand.c_str());

	// Parse output file
	ifstream preds(predictions.c_str());

	// Convert to integer
	preds>>predClassIndex;

	//cout << classList[predClassIndex-1] << endl;

	// Return class name
	return classList[predClassIndex-1];
  }


  // void classify(const Dataset &filenames)
  // {
  //   if(isColor){
  //     outFileName = "test_svm4800.dat";
  //     convertTosvmformat(filenames,outFileName);
  //     system("./svm_multiclass_classify test_svm4800.dat svm4800model test_preds4800");
  //   }else{
  //     outFileName = "test_svm1600.dat";
  //     convertTosvmformat(filenames,outFileName);
  //     system("./svm_multiclass_classify test_svm1600.dat svm1600model test_preds1600");
  //   }	      
  // }

  virtual void load_model()
  {
  	// Do nothing
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename,bool isColor)
    {
      CImg<double> a;
      if(isColor)
        a = CImg<double>(filename.c_str()).resize(size,size,1,3).unroll('x');
      else
        a = CImg<double>(filename.c_str()).resize(size,size,1,1).unroll('x');

      return a;
    }

  static const int size=40;  // subsampled image resolution
  string modelName;
};
