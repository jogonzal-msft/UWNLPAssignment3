import edu.berkeley.nlp.Test;

public class Main {

    public static void main(String[] args) {
        System.out.println("Hello World!");
        Test<MockClass> test = new Test<MockClass>();
        test.main(new String[]{
                "-test",
                "-maxTrainLength", "40",
                "-maxTestLength", "20",
                "-usestip",
                "-quiet",
                "-verticalmarkov",
                "-horizontalmarkov"
        });
    }
}
