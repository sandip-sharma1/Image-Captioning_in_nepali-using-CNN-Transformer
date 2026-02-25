
from preprocessing import *
from Transformer import caption_model

Model_path='Saved_model'
# define loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

# early stopping criteria
early_stopping = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
# checkpoint
#checkpoint = keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True)
#no custom lrs for this training...we use only 0,001 for this




# compile model
caption_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=cross_entropy)


# fit the model
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset
    #callbacks=[early_stopping]
    )

# save model
caption_model.build(input_shape=(None, *IMAGE_SIZE))  
caption_model.save_weights(os.path.join(Model_path, 'betters.weights.h5'))
caption_model.summary()