#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

class RectRegion
{
	public:
		double height1,height2,width1,width2;
		RectRegion(){}
		void set(double _width1,double _heigth1,double _width2,double _heigth2){
			width1 = _width1;
			height1 = _heigth1;
			width2 = _width2;
			height2 = _heigth2;
		}
	
};

class Rect{

	public:
		vector<RectRegion> darkPortion;
		vector<RectRegion> lightPortion;
};


class SVM_Haar : public Classifier
{
	//File names
    string trainData;
	string model; 
	string Predictions;
	string testData; 
  	//Class names
  	vector<string> classList;

  	//Class names
    map<string,int> classIndexMap;

    //num rectangles
    int numRect;
	
public:
  ofstream outfile;

  // constructor
  SVM_Haar(const vector<string> &_class_list,int _numRect) : Classifier(_class_list) 
   {
   		numRect = _numRect;
   		classList = _class_list;
	   for(int i=0;i <_class_list.size();++i)
	   	{
			classIndexMap[_class_list[i]] = i+1;
	    }
   }
  
  // overload function 
  virtual void train(const Dataset &filenames) 
   { 
  		trainData = "SVMtrain/train_haar.dat";
		model = "SVMmodels/svm_haar_model";
		
		outfile.open(trainData.c_str());
	   	
	   	vector<Rect> rectangles;
		vector<int> classes;
	  	// Extract 500 random rectangles
	    int i = 0;
	    for(int rectIndex=0;rectIndex<numRect;rectIndex++){

			rectangles.push_back(getRect(rand()%4));

	    }

		CImg<int> VecFeatures(10,numRect,1,1,255);
		int w = 0;
		int h = 0;

		for(int i = 0;i<rectangles.size();i++){
			w = 0;
		
			//Dark portion
			for(int j = 0;j<rectangles[i].darkPortion.size();j++)
			{
		
				VecFeatures(w++,h) = rectangles[i].darkPortion[j].width1;
				VecFeatures(w++,h) = rectangles[i].darkPortion[j].height1;
				VecFeatures(w++,h) = rectangles[i].darkPortion[j].width2;
				VecFeatures(w++,h) = rectangles[i].darkPortion[j].height2;
			}
			//Grey line
			VecFeatures(w++,h) = 128;
			
			//Light portion
			for(int j = 0;j<rectangles[i].lightPortion.size();j++)
			{
				
				
				VecFeatures(w++,h) = rectangles[i].lightPortion[j].width1;
				VecFeatures(w++,h) = rectangles[i].lightPortion[j].height1;
				VecFeatures(w++,h) = rectangles[i].lightPortion[j].width2;
				VecFeatures(w++,h) = rectangles[i].lightPortion[j].height2;
			}
			
			h++;
		}
		
		//Save haar features
		VecFeatures.save_png("intermediate/haar_features.png");
			

		//Write into SVM file
		int f = 0;	
		
		CImg<double> FeatureData(numRect,getSize(filenames),1,1,0);
		
		
	    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
	    {
			cout << "Processing folder:  " << c_iter->first << endl;
			
		
			for(int i=0; i<c_iter->second.size(); i++,++f)
			{	
		
				classes.push_back(classIndexMap[c_iter->first.c_str()]);
				
		
				CImg<double> imageFeatures = extract_features(c_iter->second[i].c_str());			
		
				collectFeatures(imageFeatures,rectangles,FeatureData,f);
			}
	    }

	    FeatureData = FeatureData.normalize(-1000,1000);
		
		for(int h = 0;h < FeatureData.height(); ++h)
		{
		
			outfile << classes[h] <<" ";
			for(int w = 0;w < FeatureData.width();++w)
			{	float val = 0;
				if(!isnan(FeatureData(w,h))){
					val = FeatureData(w,h);
				}
				outfile << w + 1 << ":" << val << " ";
			}
			outfile << endl;
		}

		outfile.close();
		
		
		string svmCommand = "./svm_multiclass_learn -c 100 "+ trainData+ " " +model; 
		system(svmCommand.c_str());

   }

  // overload function
  virtual string classify(const string &filename)
  {

		Predictions = "SVMtest/test_haar.dat";
		testData = "SVMmodels/svm_haar_model";
		model = "SVMpreds/predictions_haar.dat";

		int predClassIndex;

		outfile.open(Predictions.c_str());

		CImg<double> FeatureData(1000,1,1,1,0);
		
		ofstream svmTestData(testData.c_str());
				

		CImg<double> testImage = extract_features(filename);
		CImg<int> Features("Haar/Features.png");
		 vector<Rect> features;
		  
		  for(int h=0;h<Features.height();++h)
		  {  
				if(Features(0,h) == 255)
					break;
				bool black = true;	
				Rect f;
				for(int w = 0;w<Features.width();++w)
				{
					if(Features(w,h) == 100)
					{
						black = false;
						w++;
					}
					else if(Features(w,h) == 255)
						break;
					
					RectRegion r;
					r.width1 = Features(w++,h);	
					r.height1 = Features(w++,h);
					r.width2 = Features(w++,h);
					r.height2 = Features(w,h);
					if(black)
						f.darkPortion.push_back(r);
					else
						f.lightPortion.push_back(r);
		    	}
			
			features.push_back(f);
	        }

		collectFeatures(testImage,features,FeatureData,0);
		
		for(int hIndex = 0;hIndex<FeatureData.height();hIndex++)
		{
			outfile << getClassIndex(filename, classIndexMap) << " ";
			for(int wIndex = 0; wIndex < FeatureData.width();wIndex++)
			{
				outfile << " " << wIndex + 1 << ":" << FeatureData(wIndex,hIndex);
			}
			outfile<<endl;
		}
		
		
	    string svmCommand = "./svm_multiclass_classify "+ testData+ " " +model+ " "+Predictions; 
		system(svmCommand.c_str());
		
		// Read predictions
		// Parse output file
		ifstream preds(Predictions.c_str());

		// Convert to integer
		preds>>predClassIndex;

		// Return class name
		return classList[predClassIndex - 1];
  }

  virtual void load_model()
  {
		
		  

	}


protected:

  
  CImg<double> extract_features(const string &filename)
    {
		CImg<double> a= CImg<double>(filename.c_str()).resize(size, size, 1, 1);

		// cout << a.width() << a.height() << a.depth() << endl;
		return a;
    }

  static const int size=40;  
  vector<Rect> features;
  
  double calcArea(RectRegion r,CImg<double>& sumAreaTab,CImg<double>& sumSqArea)
  {
	double A = 0;
    double B = 0;
    double C = 0;
    double D = 0;
	double areaValue;
	double mean;
	double sqrAreaValue;
	double A_S = 0;
	double B_S = 0;
	double C_S = 0;

	int sumSqStd;
	int standardDeviation;
	int a = 0;
	int b = 0;

    
    if(r.height1 != 0)
	{
        C = sumAreaTab(r.width2,r.height1-1);
		C_S = sumSqArea(r.width2,r.height1-1);
		a = r.height1-1;
    }
    if(r.width1 != 0)
	{
        B = sumAreaTab(r.width1-1,r.height2);
		B_S = sumSqArea(r.width1-1,r.height2);
		b = r.width1-1;
	}
	if(r.height1 != 0 && r.width1 != 0 )
	{
        A = sumAreaTab(r.width1-1,r.height1-1);
		A_S = sumSqArea(r.width1-1,r.height1-1);
	}
    
    D = sumAreaTab(r.width2,r.height2);
  	
	areaValue = (D-B-C+A);  
	  
	mean = areaValue/((r.width2-b) * (r.height2-a));
	  
	sqrAreaValue = (sumSqArea(r.width2,r.height2) - B_S - C_S + A_S);
	  
	sumSqStd = (int)(sqrAreaValue/((r.width2-b) * (r.height2-a))) - pow(mean,2);
	
	if(sumSqStd > 0)
	  standardDeviation = sqrt(sumSqStd);
	else
		standardDeviation = 1;
	  
	return (areaValue/standardDeviation);
	  
  }
  
  
  void collectFeatures(CImg<double>& imageFeatures,vector<Rect>& features,CImg<double>& FeatureData,int f)
  {
		
		CImg<double> sumAreaTab(imageFeatures.width(),imageFeatures.height(),1,1,0);
		sumAreaTab(0,0) = imageFeatures(0,0);
		
		for(int w = 1;w<imageFeatures.width();++w)
			sumAreaTab(w,0) = sumAreaTab(w-1,0) + imageFeatures(w,0);
		 
		for(int h = 1;h<imageFeatures.height();++h)
			sumAreaTab(0,h) = sumAreaTab(0,h-1) + imageFeatures(0,h);
		
		for(int h = 1;h<imageFeatures.height();++h)
		{
			for(int w = 1;w<imageFeatures.width();++w)
			{
				sumAreaTab(w,h) = imageFeatures(w,h) + sumAreaTab(w-1,h) + sumAreaTab(w,h-1) - sumAreaTab(w-1,h-1);
			}
		}

		
		CImg<double> sumSqArea(imageFeatures.width(),imageFeatures.height(),1,1,0);
		sumSqArea(0,0) = pow(imageFeatures(0,0),2);

		for(int w = 1;w<imageFeatures.width();++w)
			sumSqArea(w,0) = sumSqArea(w-1,0) + pow(imageFeatures(w,0),2);
		
		for(int h = 1;h<imageFeatures.height();++h)
			sumSqArea(0,h) = sumSqArea(0,h-1) + pow(imageFeatures(0,h),2);
		
		for(int w = 1;w<imageFeatures.width();++w)
		{
			for(int h = 1;h<imageFeatures.height();++h)
			{
				sumSqArea(w,h) = pow(imageFeatures(w,h),2) + sumSqArea(w-1,h) + sumSqArea(w,h-1) - sumSqArea(w-1,h-1);
			}
		}
		

		for(int i = 0;i<features.size();++i)
		{
			double val = 0;
		
			for(int j = 0;j<features[i].darkPortion.size();++j)
			{
				val+= calcArea(features[i].darkPortion[j],sumAreaTab,sumSqArea);
			}
		
			for(int j = 0;j<features[i].lightPortion.size();++j)
			{
				val-= calcArea(features[i].lightPortion[j],sumAreaTab,sumSqArea);
			}
			FeatureData(i,f) = val;				
		}
  }
  
  Rect getRect(int rndNumber)
	{

	RectRegion lRegion;
	RectRegion dRegion;

	while(true){
	
	int size = (rand() % 8) + 1;
	int lW1 = rand() % 40;
	int lH1 = rand() % 40;
	int lW2 = (lW1 + (size * 1));
	int lH2 = lH1 + (size * 4);

	if(lW2>40 || lH2>40){
		continue;
	}

	int dW1 = lW2;
	int dH1 = lH1;
	int dW2 = lW1 + (size * 2);
	int dH2 = lH2;

	if(dH2>40){
		continue;
	}
	if(rndNumber== 0){
		lRegion.set(lW1,lH1,lW2,lH2);

		dRegion.set(dW1,dH1,dW2,dH2);

	}else if(rndNumber== 1){

		dRegion.set(lW1,lH1,lW2,lH2);

		lRegion.set(dW1,dH1,dW2,dH2);
	}else if(rndNumber== 2){

		dRegion.set(lH1,lW1,lH2,lW2);

		lRegion.set(dH1,dW1,dH2,dW2);
	}else{
		lRegion.set(lH1,lW1,lH2,lW2);

		dRegion.set(dH1,dW1,dH2,dW2);
	}
	break;

	}
	
	
		Rect r;
		r.lightPortion.push_back(lRegion);
		r.darkPortion.push_back(dRegion);
		return r;  
	}

	int getSize(const Dataset &filenames) {
		int numImages=0;
    for (Dataset::const_iterator c_iter = filenames.begin();
         c_iter != filenames.end(); ++c_iter) {
      for (int i = 0; i < c_iter->second.size(); i++) numImages++;
    }

    return numImages;
  }


};


