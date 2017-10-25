int getClassIndex(string fileName,map<string,int> classIndex){

	string delim = "/";
	int pos = 0;
	int bIndex =0;
	string sstring;
	while ((pos = fileName.find(delim)) != string::npos) {
	    sstring = fileName.substr(0, pos);
	    fileName.erase(0, pos + delim.length());
	    bIndex++;
	    if(bIndex == 2)
	    	break;
	}
	return classIndex[sstring];
}