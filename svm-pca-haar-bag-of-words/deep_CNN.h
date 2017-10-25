#include <fstream>


class SVM_CNN : public Classifier
{
private:
  bool isColor;

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
  SVM_CNN(const vector<string> &_class_list,bool _isColor=true) : Classifier(_class_list) {
  	  isColor = _isColor;
  	  classList = _class_list;
  for(int i=0;i<_class_list.size();i++){
      classIndexMap[_class_list[i]] = (i+1);
    }
  }
  
  // CNN training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames){

	trainData = "SVMtrain/train_data_cnn.dat";
	model = "SVMtrain/overfeat_model";
	outfile.open(trainData.c_str());  
	
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
    {
		cout << "Processing folder: " << c_iter->first << endl;
		
		for(int i=0; i<c_iter->second.size(); i++)
		{
	        int cIndex = getClassIndex(c_iter->second[i],classIndexMap);
	        outfile<<cIndex<<" ";

			CImg<double> features = getDeepFeatures(c_iter->second[i].c_str(),isColor);

			for(int fIndex = 0;fIndex<features.width();fIndex++)
				outfile<<" "<<fIndex+1<<":"<<features(fIndex,0,0);
			outfile<<endl;
		}
    }
	outfile.close();

	string svmCommand = "./svm_multiclass_learn -c 1 "+ trainData+ " " +model; 
	system(svmCommand.c_str());
  }

  virtual string classify(const string &filename){

		int predClassIndex;

	if(isColor){
		testData = "SVMtest/test_deepCNN_color.dat";
		model = "SVMmodels/svm_deepCNN_color_model";
		predictions = "SVMpreds/preds_deepCNN_color.dat";
	}
	else{
		testData = "SVMtest/test_deepCNN_grey.dat";
		model = "SVMmodels/svm_deepCNN_grey_model";
		predictions = "SVMpreds/preds_deepCNN_grey.dat";
	}
	CImg<double> testFeatures = getDeepFeatures(filename,isColor);
    outfile.open(testData.c_str());  
	outfile << getClassIndex(filename,classIndexMap) << " ";

	for(int fIndex = 0; fIndex<testFeatures.size();fIndex++){
		outfile<<" "<<fIndex+1<<":"<<testFeatures[fIndex];
	}

	outfile<<endl;
   	cout << "features written" << endl;

	outfile.close();
	string svmCommand = "./svm_multiclass_classify "+ testData+ " " +model+ " "+predictions; 
	system(svmCommand.c_str());

	ifstream preds(predictions.c_str());
	preds>>predClassIndex;
	cout << classList[predClassIndex-1] << endl;

	return classList[predClassIndex-1];
		
  }


  virtual void load_model()
  {
      //DO nothing
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
        	// Do nothing
    }

	CImg<double> getDeepFeatures(const string &filename,bool isColor)
	{
		string tempImg = "intermediate/temp.jpg";
		string featuresFile = "intermediate/localFeatures.dat";
		CImg<double> image(filename.c_str());
		if(isColor)
			image.resize(size,size,1,3);
		else
			image.resize(size,size,1,1);

		image.save(tempImg.c_str());

		cout << "Processing file: " << filename << endl;
		string ovCommand = "./overfeat/bin/linux_64/overfeat -L 18 intermediate/" +tempImg +" > " + featuresFile;
		cout << ovCommand << endl;
		system(ovCommand.c_str());

		int x,y,z;
		
		ifstream localFeatures(featuresFile.c_str());
		localFeatures>>x>>y>>x; 

		CImg<double> features(x*y*z,1,1);
		int fIndex=0;
		double featureVal;
		while(!localFeatures.eof())
		{	
			features>>featureVal;
			features(fIndex,0,0) = featureVal;
			fIndex++;
		}
		return features;
	}
    
  static const int size=250;  // subsampled image resolution
  string modelName;
  int numImages;
};
