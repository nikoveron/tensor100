const model = tf.sequential()

async function Entrenar() {
    const repeticiones = parseInt(document.getElementById('repeticiones').value);
     const epochs = repeticiones;
  
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 
   
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  
   /*await model.fit(xs, ys, {epochs: repeticiones});

   /* console.log("termino de entrenar mi modelo") */

    /* alert("termino") */

    const history = await model.fit(xs, ys, {
        epochs: epochs,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
             console.log(logs);
             console.log("/n");
             console.log(`Epoch ${epoch+1} - Loss: ${logs.loss.toFixed(4)}`);
          }
        }
      });

      // Imprimir la pérdida final
      console.log(`Final Loss: ${history.history.loss[epochs-1].toFixed(4)}`);
     
      alert("termino de entrenar");
}


async function Predecir() {
    const prediccionValor = parseInt(document.getElementById('valorPredecir').value);

    document.getElementById('micro-out-div').innerText =
    model.predict(tf.tensor2d([prediccionValor], [1, 1])).dataSync();
}
