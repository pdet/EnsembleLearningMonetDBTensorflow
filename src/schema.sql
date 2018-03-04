START TRANSACTION;
CREATE TABLE image_class (
    id integer NOT NULL PRIMARY KEY,
    label varchar(30) NOT NULL
);

CREATE TABLE image_superclass (
    id integer NOT NULL PRIMARY KEY,
    label varchar(30) NOT NULL
);


CREATE TABLE cifar100 (
    data BLOB NOT NULL,
    class INTEGER NOT NULL,
    superclass INTEGER NOT NULL,
    train BOOLEAN NOT NULL
);


CREATE TABLE classificationmodel (
    name varchar(120) NOT NULL,
    model_path varchar(120) NOT NULL, #Future BLOB
    batch_size integer NOT NULL ,
    learning_rate float NOT NULL, 
    epoch integer NOT NULL, 
    image_superclass_id integer NOT NULL
);

COMMIT;
