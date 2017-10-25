#include <fstream>

// SVM Class
class SVM_Kmeans : public Classifier
{
private:

  // Number of clusters
  int nClusters;

  // Number of kmeans iteration
  int n_itr;

  //Class index mapping
  map<string,int> classIndexMap;

  //Class names
  vector<string> classList;

  //File names
  string trainData;
  string model;
  string testData;
  string predictions;

public:
  ofstream outfile;
  SVM_Kmeans(const vector<string> &_class_list,int _nClusters) : Classifier(_class_list) {
  	  n_itr = 10;
      nClusters=_nClusters;
	  classList = _class_list;

	  //Create class mapping
      for(int i=0;i<_class_list.size();i++){
      	classIndexMap[_class_list[i]] = (i+1);
      }
  }
  
  // SVM training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames){

  	//File names
	trainData = "SVMtrain/train_bow.dat";
	model = "SVMmodels/bowmodel";
	cout << "Creating training file" << endl;
	outfile.open(trainData.c_str());

	//Get bag of visual words features
	map<string,vector<int> > bowModel = getFeatures(filenames);
	cout << "bow model created" << endl;

	//Write into SVM format
	for(map<string,vector<int> >::iterator c_iter=bowModel.begin(); c_iter != bowModel.end(); ++c_iter)
	{
		int cIndex = getClassIndex(c_iter->first,classIndexMap);
		outfile<<cIndex<<" ";

		for(int i=0; i<c_iter->second.size(); i++){
			outfile<<i+1<<":"<<c_iter->second[i]<<" ";		
		}
			outfile<<endl;
	}

	outfile.close();

	//Execute svm library
	string svmCommand = "./svm_multiclass_learn -c 10 "+ trainData+ " " +model;
	system(svmCommand.c_str());    
  }

  virtual string classify(const string &filename){

  	//File names
	testData = "SVMtest/test_bow.dat";
	model = "SVMmodels/bowmodel";
	predictions = "SVMpreds/predictions_bow.dat";
	int predClassIndex;

    vector<vector<float> > cList;  

    //Read train centroids
	CImg<double> final_centroid("intermediate/tr_kmeans_centroids.png");
	for(int i=0; i<final_centroid.height();i++){
	  vector<float> temp(128);
	  for(int j=0; j<final_centroid.width();j++){
	    temp[j] = final_centroid(j,i,0,0);
	  }
          cList.push_back(temp);
	}
	// cout << "Centroids ready" << endl;
	outfile.open(testData.c_str());

	// cout <<"class index" << getclassIndexMap(filename) <<endl;
	outfile << getClassIndex(filename,classIndexMap) << " ";
	
	vector<int> testFeatures(cList.size(),0);
    CImg<double> testImg(filename.c_str());
    testImg.resize(size,size,1,3);

    vector<SiftDescriptor> sDec = applySift(testImg);
    for(int i = 0; i<sDec.size();i++){

        // Initialize minimum distance
        double minDist = INT_MAX;
        int cIndex;
       
        //Calc euclidean distance between sift descriptor and centroids
        for(int j=0; j<cList.size(); j++){
          double dist = calcEuclideanDist(cList[j], sDec[i].descriptor);
          if(dist < minDist){
            cIndex = j;
            minDist = dist;   
          }
        }

        // Update number of points in cluster
        testFeatures[cIndex] += 1;
       }

   	cout << "features ready" << endl;
	for(int fIndex = 0; fIndex<cList.size();fIndex++){
		cout << testFeatures[fIndex] << " ";
		outfile<<" "<<fIndex+1<<":"<<testFeatures[fIndex];
	}
		
	outfile<<endl;
   	cout << "features written" << endl;

	outfile.close();
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

    map<string,vector<int> > getFeatures(const Dataset &filenames){
    	
    	vector<vector<float> > cList;  
		map<string,vector<int> > bowModel;
	    map<string, vector<SiftDescriptor> > descriptorMap;
		vector<double > numPointsCluster;

	    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter){

      		cout << "Processing folder: " << c_iter->first << endl;
	    	// 1. Get sift discriptor
		 	for(int i=0;i< c_iter->second.size();i++){
				CImg<double> input(c_iter->second[i].c_str());	  
				input.resize(size,size,1,3);
				vector<SiftDescriptor> sDesc = applySift(input);

				descriptorMap[c_iter->second[i]] = sDesc;
			}

			// 2. Initialize cluster centroids
			for(int cIndex=0; cIndex< nClusters; cIndex++){  
				//Randomly pick any image as cluster
				int cId = getRndIndex(c_iter->second.size());
				CImg<double> rndCentroid(c_iter->second[cId].c_str());
				rndCentroid.resize(size,size,1,3);

				numPointsCluster.push_back(0);
				//Pick random sift descriptor of random image as centroid
				vector<SiftDescriptor> sDesc = Sift::compute_sift(rndCentroid);
				cList.push_back(sDesc[getRndIndex(sDesc.size())].descriptor);
			}	

		}
		// 3. Apply kmeans algorithm
		for(int itrIndex=0; itrIndex<n_itr; itrIndex++){
			cout << "kmeans iteration: " <<itrIndex << endl;
			kmeans(descriptorMap,cList,numPointsCluster);
		}

		//4. Save final centroid
		CImg<double> final_centroid(128,cList.size(),1,1,0);
		for(int i=0; i<cList.size();i++){
			for(int j=0; j<128;j++){
				cout << cList[i][j] << " ";
				final_centroid(j,i,0,0) = cList[i][j];
			}
		}

		final_centroid.save("intermediate/tr_kmeans_centroids.png"); 

		//Transform each image to 500 visual words 
		for(map<string, vector<SiftDescriptor> >::iterator it=descriptorMap.begin(); it!=descriptorMap.end();++it){
			vector<int> features;

			//Initalize feature vector with 500 dimensions
			for(int fIndex=0;fIndex<cList.size();fIndex++){
				features.push_back(fIndex);
			}

			//Create histogram of each image
			for(int dIndex=0; dIndex<it->second.size();dIndex++){
				features[it->second[dIndex].col] += 1;
			}
				bowModel[it->first] = features;
		}  

		return bowModel;
    }

    void kmeans(map<string, vector<SiftDescriptor> > &descriptorMap,vector< vector<float> > &cList, vector<double > &numPointsCluster){

    //Part 1: Cluster Assignments
    for(map<string, vector<SiftDescriptor> >::iterator it=descriptorMap.begin(); it!=descriptorMap.end();it++){
      
      for(int i = 0; i<it->second.size();i++){

        // Initialize minimum distance
        double minDist = INT_MAX;
        int cIndex;
        vector<float>  sPoints = it->second[i].descriptor;
        
        //Calc euclidean distance between sift descriptor and centroids
        for(int j=0; j<cList.size(); j++){
          double dist = calcEuclideanDist(cList[j], sPoints);
          if(dist < minDist){
            cIndex = j;
            minDist = dist;   
          }
        }

        // Assign cluster with min distance to discriptor
        vector<SiftDescriptor> sDiscriptor = descriptorMap[it->first];
        sDiscriptor[i].col = cIndex;
        descriptorMap[it->first] = sDiscriptor;

        // Update number of points in cluster
        numPointsCluster[cIndex] += 1;
       } 
     }

    //Part 2: Update cluster centroids
    for(map<string, vector<SiftDescriptor> >::iterator it=descriptorMap.begin(); it!=descriptorMap.end();++it){
      for(int i=0; i <it->second.size();i++){

        //Get cluster assignment
        int cIndex = it->second[i].col;

        //Add this descriptor to cluster
        for(int j=0; j<128; j++){
          cList[cIndex][j] += it->second[i].descriptor[j]/numPointsCluster[cIndex];
        }
      }
    }

  }

    vector<SiftDescriptor> applySift(CImg<double> input){
		CImg<double> greyImg = input.get_RGBtoHSI().get_channel(2);
		vector<SiftDescriptor> sDesc = Sift::compute_sift(greyImg);
		return sDesc;
    }

    int getRndIndex(int _size){
    	return rand() % _size;
    }

    double calcEuclideanDist(vector<float> point1, vector<float> point2){
		double dist = 0;
		for(int i =0; i < point1.size(); i++){
			dist += pow(( point1[i] - point2[i]),2) ;
		}
		return sqrt(dist);
	}

    
  static const int size=140;  // subsampled image resolution
  string modelName;
  int numImages;
};
