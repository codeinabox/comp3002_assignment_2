CXX = g++
CXXFLAGS = -Wall -Wcast-qual -g 
APP = ass2
OBJS = NeuralNet.o ass2.o

$(APP): ${OBJS}
	g++ ${CXXFLAGS} ${OBJS} -o ${APP}

clean:
	rm -f *.o

allclean: clean
	rm -f ${APP}

