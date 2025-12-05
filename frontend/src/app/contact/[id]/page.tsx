"use client";

import { useParams } from "next/navigation";

export default function ContactIdPage() {
  const params = useParams();
  const { id } = params;

  return (
    <div>
      <h1>{id}</h1>
    </div>
  );
}