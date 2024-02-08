package NeuralNetwork;

import java.util.List;
import java.util.Map;

public class NeuralNerwork extends BaseObject{
    public static final String INPUT_HIDDEN = "INPUT_HIDDEN";
    public static final String HIDDEN_OUTPUT = "HIDDEN_OUTPUT";
    public static final String HIDDEN_BIAS = "HIDDEN_BIAS";
    public static final String OUTPUT_BIAS = "OUTPUT_BIAS";

    Map<String, Matrix> matMap;
    double learningRate = 0.01;

    public NeuralNerwork(int i, int h, int o){
        matMap.put(INPUT_HIDDEN , new Matrix(h, i));
        matMap.put(HIDDEN_OUTPUT, new Matrix(o, h));
        matMap.put(HIDDEN_BIAS  , new Matrix(h, 1));
        matMap.put(HIDDEN_OUTPUT, new Matrix(o, 1));
    }

    /**
     * 
     * @param x
     * @return
     * @throws Exception
     */
    public List<Double> predict(double[] x) throws Exception{
        Matrix input = Matrix.fromArray(x);
        Matrix hidden = Matrix.dotProduct(matMap.get(INPUT_HIDDEN), input);
        hidden.add(matMap.get(HIDDEN_BIAS));
        hidden.sigmoid();

        Matrix output = Matrix.dotProduct(matMap.get(HIDDEN_OUTPUT),hidden);
        output.add(matMap.get(OUTPUT_BIAS));
        output.sigmoid();
        
        return output.toArray();
    }
}
