package src;

import java.io.*;

import weka.classifiers.functions.*;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instances;

public class normalInput {
    public static void main(String[] args) {
        try {
            String pathToWholeData = "dataSets/3months_100aggregate_extraFeatures.arff";
            //String pathToHugeData = "/home/cycle/workspace/wekaforecasting-new-features/dataSets/6months_1aggregate_extraFeatures.arff";

            // load the data
            Instances wholeData  = new Instances(new BufferedReader(new FileReader(pathToWholeData)));
            //Instances hugeData = new Instances(new BufferedReader((new FileReader(pathToHugeData))));

            wholeData.setClassIndex(1);
            //hugeData.setClassIndex(1);
            //select_Attributes(airlineData);
            //AttributeSelectedClassifier attributeSelectedClassifier = applyMetaClassifier.applyMetaClassifier(airlineData);
            AttributeSelectedClassifier attributeSelectedClassifier2 = new AttributeSelectedClassifier();
            IBk iBk = new IBk();
            LinearRegression linearRegression = new LinearRegression();
            linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1 -R 1E-6"));
            MLPRegressor mlpRegressor = new MLPRegressor();
            //mlpRegressor.setOptions(weka.core.Utils.splitOptions("-N 1"));
            doForecasting doForecasting = new doForecasting();
            doForecasting.doForecast(wholeData, linearRegression);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
