package src;

import weka.attributeSelection.*;
import weka.classifiers.functions.*;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instances;

/**
 * Created by cycle on 09.12.16.
 */
public class applyMetaClassifier {
    public static AttributeSelectedClassifier applyMetaClassifier(Instances trainData) {
        try {
            AttributeSelectedClassifier attributeSelectedClassifier = new AttributeSelectedClassifier();
            WrapperSubsetEval wrapperSubsetEval = new WrapperSubsetEval();
            //CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
            //HoltWinters holtWinters = new HoltWinters();
            ReliefFAttributeEval reliefFAttributeEval = new ReliefFAttributeEval();
            Ranker ranker = new Ranker();
            //reliefFAttributeEval.setOptions(weka.core.Utils.splitOptions("-K 1"));
            //ranker.setOptions(weka.core.Utils.splitOptions("-N 387"));
            LinearRegression linearRegression = new LinearRegression();
            SimpleLinearRegression simpleLinearRegression = new SimpleLinearRegression();
            MLPRegressor mlpRegressor = new MLPRegressor();
            //mlpRegressor.buildClassifier(trainData);
            //SymmetricalUncertAttributeEval symmetricalUncertAttributeEval = new SymmetricalUncertAttributeEval();
            //attributeSelectedClassifier.setOptions(new String[]{"-D"});
            //mlpRegressor.setOptions(weka.core.Utils.splitOptions("-N 1"));
            linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1"));
            //wrapperSubsetEval.setOptions(weka.core.Utils.splitOptions("-F 0 -B weka.classifiers.functions.MLPRegressor"));
            mlpRegressor.setOptions(weka.core.Utils.splitOptions("-N 1"));
            wrapperSubsetEval.setClassifier(linearRegression);
            //wrapperSubsetEval.buildEvaluator(trainData);
            BestFirst bestFirstsearch = new BestFirst();
            MySearch mySearch = new MySearch();
            mySearch.setSearchTermination(96);
            //search.setSearchBackwards(true);
            //ranker.setStartSet("11, 19, 12, 13, 19");

          //     ranker1.setStartSet("1-20");
            attributeSelectedClassifier.setSearch(mySearch);
            attributeSelectedClassifier.setEvaluator(wrapperSubsetEval);
            attributeSelectedClassifier.setClassifier(mlpRegressor);

            return attributeSelectedClassifier;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}


/* AttributeSelectedClassifier attributeSelectedClassifier = new AttributeSelectedClassifier();
           Ranker ranker1 = new Ranker();
           ReliefFAttributeEval reliefFAttributeEval1 = new ReliefFAttributeEval();*/

           /* System.out.println(ranker.search(reliefFAttributeEval, trainData));
            ReliefFAttributeEval reliefFAttributeEval1 = new ReliefFAttributeEval();
            System.out.println(ranker1.search(reliefFAttributeEval1, trainData));*/