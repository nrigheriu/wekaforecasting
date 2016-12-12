package src;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MLPRegressor;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
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
            //PrincipalComponents principalComponents = new PrincipalComponents();
            SMOreg smOreg = new SMOreg();
            Ranker ranker = new Ranker();                                                                               //TODO:automatically extract first x ranked attributes
            //reliefFAttributeEval.setOptions(weka.core.Utils.splitOptions("-K 1"));
            //ranker.setOptions(weka.core.Utils.splitOptions("-N 387"));
            //reliefFAttributeEval.listOptions();
            LinearRegression linearRegression = new LinearRegression();
            MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
            MLPRegressor mlpRegressor = new MLPRegressor();
            //mlpRegressor.buildClassifier(trainData);
            //SymmetricalUncertAttributeEval symmetricalUncertAttributeEval = new SymmetricalUncertAttributeEval();
            //attributeSelectedClassifier.setOptions(new String[]{"-D"});
            linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1"));
            //linearRegression.buildClassifier(trainData);
            //wrapperSubsetEval.setOptions(weka.core.Utils.splitOptions("-F 0 -B weka.classifiers.functions.MLPRegressor"));
            //wrapperSubsetEval.setClassifier(linearRegression);
            //wrapperSubsetEval.buildEvaluator(trainData);
            BestFirst bestFirstsearch = new BestFirst();
            //search.setSearchBackwards(true);
            attributeSelectedClassifier.setSearch(ranker);
            attributeSelectedClassifier.setEvaluator(reliefFAttributeEval);
            attributeSelectedClassifier.setClassifier(linearRegression);

            return attributeSelectedClassifier;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
