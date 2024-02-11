package NeuralNetwork;

public enum Operation {
    PLUS {
        @Override
        public int calc(final int i1, final int i2) {
            return i1 + i2;
        }
    },
    MINUS {
        @Override
        public int calc(final int i1, final int i2) {
            return i1 - i2;
        }
    },
    MULTIPLY {
        @Override
        public int calc(final int i1, final int i2) {
            return i1 * i2;
        }
    },
    DIVIDE {
        @Override
        public int calc(final int i1, final int i2) {
            return i1 / i2;
        }
    };

    public abstract int calc(int i1, int i2);
}