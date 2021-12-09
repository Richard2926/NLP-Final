export default function Rating(props) {

  return (
    <button class={(props.rating === props.value ? 
        "box-border h-32 w-32 p-4 bg-white border-4 mx-4" : "box-border h-32 w-32 p-4 border-4 mx-4")} onClick={() => props.updateRating(props.value)}>{props.text}</button>
    )
}
